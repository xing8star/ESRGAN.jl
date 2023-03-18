using Optimisers,Zygote
using MLUtils

include("lux_model.jl")
include("config.jl")
mutable struct lr_scheduler
    opt::Union{NamedTuple,Tuple}
    decay_iter::Vector
    lr::Float64
    current_lr::Int
end

function lr_scheduler!(opt::lr_scheduler,current_epoch::Int)
    c=opt.current_lr
    if current_epoch<=1
        Optimisers.adjust!(opt.opt,opt.lr)
        opt.current_lr=1
    elseif c<=length(opt.decay_iter) && current_epoch>opt.decay_iter[c]
        opt.current_lr+=1
        Optimisers.adjust!(opt.opt,opt.lr*0.1^opt.current_lr)
    end
end
create_optimiser(ps;η=lr,β=(b1,b2),γ=weight_decay)=Optimisers.setup(Optimisers.AdamW(η,β,γ), ps)
using Statistics:mean

# load model
using BSON: @load

generator=ESRGAN(3,3,64)
discriminator=Discriminator()
if isfile(modelpath)
    @load modelpath st_generator st_discriminator ps_discriminator ps_generator
else
    ps_generator,st_generator=Lux.setup(rng,generator);
    ps_discriminator,st_discriminator=Lux.setup(rng,discriminator);
end
#
include("datasets.jl")
include("losses.jl")
#train tool

data_loader = get_data(data_dir,batch_size,image_size)
# data_loader = get_data("datasets/hr",2)

adversarial_criterion = logitbinarycrossentropy
content_criterion = l1_loss_mae
perception_criterion = perceptualloss

ps_generator=Lux.trainmode(ps_generator);
ps_discriminator=Lux.trainmode(ps_discriminator);
#
using BSON: @save
function savemodel(path::String)
    @save path*".bson" st_generator=cpu(st_generator) st_discriminator=cpu(st_discriminator) ps_discriminator = cpu(ps_discriminator) ps_generator= cpu(ps_generator)
end
if Lux.CUDA.functional()
    st_generator=gpu(st_generator);
    st_discriminator=gpu(st_discriminator);
    ps_discriminator = gpu(ps_discriminator);
    ps_generator= gpu(ps_generator);
    # data_loader=Flux.CUDA.CuIterator(data_loader)
end
optimizer_generator = create_optimiser(ps_generator);
optimizer_generator=lr_scheduler(optimizer_generator,decay_iter,lr,1);

optimizer_discriminator = create_optimiser(ps_discriminator);
optimizer_discriminator=lr_scheduler(optimizer_discriminator,decay_iter,lr,1);

# low_resolution=rand(Float32,128,128,3,1) |> device
# high_resolution=rand(Float32,512,512,3,2)|> device

# real_labels = ones(Float32, 1,size(high_resolution,4)) |> device
# fake_labels = zeros(Float32, 1,size(high_resolution,4))|> device

# (loss,fake_high_resolution,st_generator), back = Zygote.pullback(ps->generator_loss(low_resolution,high_resolution,fake_labels,real_labels,ps,st_generator,ps_discriminator,st_discriminator),ps_generator) 
# Lux.apply(generator,low_resolution,ps_generator,st_generator)

# gs = back((one(loss), nothing,nothing))[1]
# optimizer_generator.opt, ps_generator = Optimisers.update(optimizer_generator.opt, ps_generator, gs)
# (loss,st_discriminator), back = Zygote.pullback(ps->discriminator_loss(high_resolution,fake_high_resolution,fake_labels,real_labels,ps,st_discriminator),ps_discriminator) 
#     gs = back((one(loss), nothing))[1]

function generator_loss(low_resolution,high_resolution,fake_labels,real_labels,ps,st,ps_discriminator,st_discriminator)
    fake_high_resolution,st_generator = Lux.apply(generator,low_resolution,ps,st)

    score_real,st_discriminator = Lux.apply(discriminator,high_resolution,ps_discriminator,st_discriminator)
    score_fake,st_discriminator = Lux.apply(discriminator,fake_high_resolution,ps_discriminator,st_discriminator)
    discriminator_rf = score_real .- mean(score_fake)
    discriminator_fr = score_fake .- mean(score_real)
    if adversarial_loss_factor>0
        adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
        adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
        adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
    else
        adversarial_loss=0
    end

    if perceptual_loss_factor>0
        perceptual_loss = perception_criterion(high_resolution, fake_high_resolution)
    else
        perceptual_loss=0
    end
    content_loss = content_criterion(fake_high_resolution, high_resolution)

    generator_loss = content_loss * content_loss_factor+
                    adversarial_loss * adversarial_loss_factor + 
                    perceptual_loss * perceptual_loss_factor          
    generator_loss,fake_high_resolution,st_generator
end

function discriminator_loss(high_resolution,fake_high_resolution,fake_labels,real_labels,ps,st)
    score_real,st_discriminator = discriminator(high_resolution,ps,st)
    score_fake,st_discriminator = discriminator(fake_high_resolution,ps,st)
    discriminator_rf = score_real .- mean(score_fake)
    discriminator_fr = score_fake .- mean(score_real)

    adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
    adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
    discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2
    discriminator_loss,st_discriminator

end
using ProgressMeter: Progress, next!

function trainesrgan(epochs::Int,data_loader)
    global ps_discriminator,ps_generator,st_discriminator,st_generator
    global modelname
    for epoch in 1:epochs
        steps=1
        progress = Progress(length(data_loader))
        for (step, (low_resolution,high_resolution)) in enumerate(data_loader)
            # low_resolution,high_resolution=low_resolution|>device,high_resolution|> device
            real_labels = ones(Float32, 1,size(high_resolution,4)) |> device
            fake_labels = zeros(Float32, 1,size(high_resolution,4))|> device
            
            ##########################
            #   training generator   #
            ##########################
            # fake_high_resolution,st_generator = generator(low_resolution,ps_generator,st_generator)
            
            (gloss,fake_high_resolution,st_generator), back = Zygote.pullback(ps->generator_loss(low_resolution,high_resolution,fake_labels,real_labels,ps,st_generator,ps_discriminator,st_discriminator),ps_generator) 
            
            gs = back((one(gloss), nothing,nothing))[1]
            optimizer_generator.opt, ps_generator = Optimisers.update(optimizer_generator.opt, ps_generator, gs)
        
            ##########################
            # training discriminator #
            ##########################
            (dloss,st_discriminator), back = Zygote.pullback(ps->discriminator_loss(high_resolution,fake_high_resolution,fake_labels,real_labels,ps,st_discriminator),ps_discriminator) 
            gs = back((one(dloss), nothing))[1]
            optimizer_discriminator.opt, ps_discriminator = Optimisers.update(optimizer_discriminator.opt, ps_discriminator, gs)
            next!(progress; showvalues=[(:generator_loss, gloss),(:discriminator_loss,dloss)])
            lr_scheduler!(optimizer_generator,steps)
            lr_scheduler!(optimizer_discriminator,steps)
            steps+=1
            if step % 50 == 0
                result = cat(high_resolution, fake_high_resolution, dims=2)|>cpu
                save_safe_image(result, joinpath(sample_dir, string(epoch), "SR_$step.png"))
                
            end
        end
        savemodel(modelname)
    end
end

trainesrgan(num_epoch,Lux.CUDA.CuIterator(data_loader))
# trainesrgan(num_epoch,data_loader)
# save_safe_image(fake_high_resolution, joinpath(sample_dir, string(1), "SR_1.png"))
# model = loadmodel!(model, @load("mymodel.bson"))
savemodel(modelname)
