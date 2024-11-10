GPU=1
LR=0.01
Epoch=100
Versions=(fan mlp)
Dataset=MNIST #(MNIST MNIST-M Fashion-MNIST Fashion-MNIST-corrupted)
logdirpath=result

if [ ! -d ./${logdirpath} ]; then
    mkdir ./${logdirpath}
fi

for Version in "${Versions[@]}"; do
    path=${Version}

    echo "正在运行 ${path}..."
    python3 -u ./test_image_recognition.py \
    --gpu_id ${GPU} \
    --lr ${LR} \
    --epoch ${Epoch} \
    --version ${Version} \
    --dataset ${Dataset} \
    > ./${logdirpath}/${Dataset}_${path}.log 2>&1 &
done

wait

