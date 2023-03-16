using Lux
using Lux.NNlib
import Random

function ConvBlock(inc::T,out::T,k,s,p,use_act) where T<:Int
    if use_act 
        return Conv((k,k),inc => out,x -> leakyrelu.(x,0.2),stride = s,pad = p,bias=true)
    else
        return Conv((k,k),inc => out,stride = s,pad = p,bias=true)
    end
end
# {C1,C2,C3,C4,C5}
struct ResidualDenseBlock <: Lux.AbstractExplicitContainerLayer{tuple([Symbol("layer"*string(i)) for i in 1:5]...)}
    residual_beta::AbstractFloat
    # layer1::C1
    # layer2::C2
    # layer3::C3
    # layer4::C4
    # layer5::C5
    layer1::Conv
    layer2::Conv
    layer3::Conv
    layer4::Conv
    layer5::Conv
end

function ResidualDenseBlock(nf,gc=32, res_scale=0.2f0)
    blocks = []
    for i in 0:4
        in_channels = nf + gc * i
        out_channels = i<=3 ? gc : nf
        use_act = i<=3 ? true : false
        push!(blocks,ConvBlock(in_channels,out_channels,3,1,1,use_act))
    end

    return ResidualDenseBlock(res_scale,blocks...)
end


# namedtuple2vector(obj)=[getfield(obj,Symbol("layer"*string(i))) for i in 1:length(fieldnames(typeof(obj)))]
function (m::ResidualDenseBlock)(x,ps,st) 
    # new_inputs = x
    # local out,new_inputs
    # blocks=[getfield(m,Symbol("layer"*string(i))) for i in 1:length(fieldnames(typeof(m)))-1]
    # ps=namedtuple2vector(ps)
    # stv=namedtuple2vector(st)
    # for (idx,block) in enumerate(blocks)
    #     out,_ = block(new_inputs,ps[idx],stv[idx])
    #     new_inputs = cat(new_inputs,out,dims=3)
    # end
    out1,st1 = m.layer1(x,ps.layer1,st.layer1)
    out2,st2 = m.layer2(cat(x,out1,dims=3),ps.layer2,st.layer2)
    out3,st3 = m.layer3(cat(x,out1,out2,dims=3),ps.layer3,st.layer3)
    out4,st4 = m.layer4(cat(x,out1,out2,out3,dims=3),ps.layer4,st.layer4)
    out5,st5 = m.layer5(cat(x,out1,out2,out3,out4,dims=3),ps.layer5,st.layer5)
    st = merge(st, (layer1=st1, layer2=st2,layer3=st3, layer4=st4, layer5=st5))
    return out5 * m.residual_beta + x,st
end


struct ResidualInResidualDenseBlock{RRDB <: Chain} <: Lux.AbstractExplicitContainerLayer{(:rrdb,)}
    residual_beta::AbstractFloat
    rrdb::RRDB
end
function ResidualInResidualDenseBlock(nf;gc=32, res_scale=0.2f0)
    rrdb = Chain([ResidualDenseBlock(nf,gc) for _ in 1:3]...)
    ResidualInResidualDenseBlock(res_scale,rrdb)
end
function (m::ResidualInResidualDenseBlock)(x,ps,st)
    out,st=m.rrdb(x,ps,st)
    out*m.residual_beta + x,st
end
function UpsampleBlock(nf,scale_factor = 2)
    return Chain(
        Upsample(:nearest,scale = (scale_factor,scale_factor)),
        Conv((3,3),nf=>nf,x -> leakyrelu.(x,0.2),stride = 1,pad = 1,bias=true)
        
    )
end

struct Generator{Initial <: Lux.AbstractExplicitLayer,Res <: Lux.AbstractExplicitLayer, 
    C <: Lux.AbstractExplicitLayer, Ups<: Lux.AbstractExplicitLayer,Fin<: Lux.AbstractExplicitLayer} <: 
       Lux.AbstractExplicitContainerLayer{(:initial,:residuals,:conv,:upsamples,:final)}
    initial::Initial
    residuals::Res
    conv::C
    upsamples::Ups
    final::Fin
end

ReflectionPad2d(pad::Int)=x->pad_reflect(x,(pad,pad,pad,pad))

function ESRGAN(in_channels, out_channels, nf=64, gc=32, scale_factor=4, n_basic_block=23)
    initial = Chain(ReflectionPad2d(1), Conv((3,3),in_channels=>nf,relu))

    residuals = Chain([ResidualInResidualDenseBlock(nf;gc) for _ in 1:n_basic_block]...)
    conv = Chain(ReflectionPad2d(1), Conv((3,3),nf=>nf,relu))
    upsamples = Chain(UpsampleBlock(nf),UpsampleBlock(nf))
    final=Chain(ReflectionPad2d(1),
        Conv((3,3),nf=>nf,x -> leakyrelu.(x,0.2)),
        ReflectionPad2d(1),
        Conv((3,3),nf=>out_channels,x -> leakyrelu.(x,0.2))
    )
    Generator(initial,residuals,conv,upsamples,final)
end
function (m::Generator)(x,ps,st)
    initial,st_initial = m.initial(x,ps.initial,st.initial)
    x1,st_residuals=m.residuals(initial,ps.residuals,st.residuals)
    x,st_conv = m.conv(x1,ps.conv,st.conv)
    x+= initial
    x,st_upsamples = m.upsamples(x,ps.upsamples,st.upsamples)
    x,st_final = m.final(x,ps.final,st.final)
    st = merge(st, (initial=st_initial, residuals=st_residuals,conv=st_conv, upsamples=st_upsamples, final=st_final))
    return x,st
end



struct Discriminator{B <: Lux.AbstractExplicitLayer,C <: Lux.AbstractExplicitLayer} <: Lux.AbstractExplicitContainerLayer{(:blocks,:classifier)}
    blocks::B
    classifier::C
end

function Discriminator(;in_channels = 3,out_channels = 64,num_conv_block=4)
    blocks = Vector()
    for _ in 1:num_conv_block
        push!(blocks,[ReflectionPad2d(1),
                    Conv((3,3),in_channels => out_channels,leakyrelu),
                    BatchNorm(out_channels)]...)
        in_channels = out_channels
        push!(blocks,[ReflectionPad2d(1),
                    Conv((3,3),in_channels => out_channels,leakyrelu;stride=2)
                    ]...)
        out_channels *= 2
    end
    out_channels =fld(out_channels,2)
    in_channels = out_channels

    push!(blocks,[Conv((3,3),in_channels => out_channels,
                x -> leakyrelu.(x,0.2)),
                Conv((3,3),out_channels=>in_channels)]...)
    blocks = Chain(blocks...)
    classifier = Chain(
        AdaptiveMeanPool((6, 6)),
        FlattenLayer(),
        Dense(512 * 6 * 6, 1024),
        Dense(1024,1)
    )
    Discriminator(blocks,classifier)
end

function (m::Discriminator)(x,ps,st)
    x,st_blocks = m.blocks(x,ps.blocks,st.blocks)
    x,st_classifier = m.classifier(x,ps.classifier,st.classifier)
    st = merge(st, (blocks=st_blocks, classifier=st_classifier))
    
    return x,st
end
