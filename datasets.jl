using Images
save_image(data,path::String)=Images.save(path,colorview(RGB,PermutedDimsArray(data[:,:,:,1],(3,1,2))))
save_safe_image(data,path::String)=Images.save(path,map(clamp01nan,colorview(RGB,PermutedDimsArray(data[:,:,:,1],(3,1,2)))))
load_image(path::String,size::Tuple)=Images.load(path) |>
                            x->imresize(x,size)|>
                            channelview |>
                            x->PermutedDimsArray(x,(2,3,1))
                            
function cite3channel(x::Array{<:AbstractFloat,4})
    if size(x,3)==4
        return x[:,:,1:3]
    else
        return x
    end
end    

function get_data(path::String,batchsize::Int,image_size::Int)
    paths=readdir(path,join=true)
    hr_image_size=image_size*4
    handle2dataset(x)=channelview(x) |>
                    x->PermutedDimsArray(x,(2,3,1)).|> 
                    cite3channel |> batch .|> Float32
    all_images=Images.load.(paths)
    lr=all_images|>x->imresize(x,(image_size,image_size)) |> handle2dataset
    hr=all_images|>x->imresize(x,(hr_image_size,hr_image_size)) |> handle2dataset
    # lr=broadcast(x->load_image(x,(image_size,image_size)),paths) .|> cite3channel |> batch .|> Float32
    # hr=broadcast(x->load_image(x,(hr_image_size,hr_image_size)),paths) .|> cite3channel |> batch .|> Float32
    DataLoader((lr, hr), batchsize=batchsize, shuffle=true,collate=true)
end
# data_loader = get_data("datasets/hr",batch_size)
