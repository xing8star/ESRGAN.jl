using Images
display_image(data)=colorview(RGB,PermutedDimsArray(vcat(unbatch(data)...),(3,1,2)))
save_image(data,path::String)=Images.save(path,display_image(data))
save_safe_image(data,path::String)=Images.save(path,map(clamp01nan,display_image(data)))
load_image(path::String)=Images.load(path) |>
                            channelview |>
                            x->PermutedDimsArray(x,(2,3,1))
load_image(path::String,size::Tuple)=Images.load(path) |>
                            x->imresize(x,size)|>
                            channelview |>
                            x->PermutedDimsArray(x,(2,3,1))
test_image(path::String)= load_image(path)|> x->unsqueeze(x,4)
function cite3channel(x::AbstractArray{<:Real,3})
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
                x->PermutedDimsArray(x,(2,3,1)) |> 
                cite3channel .|> Float32
    all_images=Images.load.(paths)
    lr=(all_images .|>x->imresize(x,(image_size,image_size)) |> handle2dataset) |>batch
    hr=(all_images .|>x->imresize(x,(hr_image_size,hr_image_size)) |> handle2dataset) |>batch
    # lr=broadcast(x->load_image(x,(image_size,image_size)),paths) .|> cite3channel |> batch .|> Float32
    # hr=broadcast(x->load_image(x,(hr_image_size,hr_image_size)),paths) .|> cite3channel |> batch .|> Float32
    DataLoader((lr, hr), batchsize=batchsize, shuffle=true,collate=true)
end
