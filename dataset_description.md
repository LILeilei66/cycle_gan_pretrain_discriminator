### horse_zebra Dataset

    len(horse.real) :   1187
    len(horse.fake) :   1474
    len(zebra.real) :   1474
    len(zebra.fake) :   1187
    
    img.shape = [256, 256, 3]
    
    batch.shape = [batch_size, channel, shape0, shape1]
    # (channel, shape, shape) 的排列顺序是 Compose.ToTensor() 的特点

