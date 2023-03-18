rng=Random.default_rng()
device = gpu
num_epoch = 100
epoch = 0
image_size = 128
batch_size = 2
data_dir="data/DIV2K_train_HR"
sample_dir = "samples"
modelname="mymodel1"
modelpath=modelname * ".bson"
nf = 32
scale_factor = 4
is_perceptual_oriented=true
b1=0.9
b2=0.999
weight_decay=1e-2
# if is_perceptual_oriented
#     lr = 2e-4
#     content_loss_factor = 1
#     perceptual_loss_factor = 0
#     adversarial_loss_factor = 0
#     decay_iter = [2e5, 2 * 2e5, 3 * 2e5, 4 * 2e5, 5 * 2e5]
# else
#     lr = 1e-4
#     content_loss_factor = 1e-1
#     perceptual_loss_factor = 1
#     adversarial_loss_factor = 5e-3
#     decay_iter = [50000, 100000, 200000, 300000]
# end
if is_perceptual_oriented
    lr = 2e-4
    content_loss_factor = 1
    perceptual_loss_factor = 0
    adversarial_loss_factor = 0
    decay_iter = [500, 2 * 500, 3 * 500, 4 * 500, 5 * 500]
else
    lr = 1e-4
    content_loss_factor = 1e-1
    perceptual_loss_factor = 1
    adversarial_loss_factor = 5e-3
    decay_iter = [500, 1000, 2000, 3000]
end