using Boltz

model,ps,st=vgg(:vgg19;pretrained=true);

model=Chain([model.layers...][1:20]...)

let psn=eachindex(ps)[1:20]
  global ps=(;zip(psn,ps[psn])...);
  global st=(;zip(psn,st[psn])...);
end
function _check_sizes(ŷ::AbstractArray, y::AbstractArray)
    for d in 1:max(ndims(ŷ), ndims(y)) 
     size(ŷ,d) == size(y,d) || throw(DimensionMismatch(
        "loss function expects size(ŷ) = $(size(ŷ)) to match size(y) = $(size(y))"
      ))
    end
  end
function logitbinarycrossentropy(ŷ, y; agg = mean)
    _check_sizes(ŷ, y)
    agg(@.((1 - y) * ŷ - logσ(ŷ)))
end
function l1_loss_mae(ŷ, y; agg = mean)
    _check_sizes(ŷ, y)
    agg(abs.(ŷ .- y))
end
loss_network(x::Array)= Lux.apply(model,x,ps,st)[1]
Zygote.ChainRulesCore.@non_differentiable loss_network(x)

function perceptualloss(high_resolution, fake_high_resolution)
  l1_loss_mae(loss_network(high_resolution), loss_network(fake_high_resolution))
end


