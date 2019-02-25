import matplotlib.image as mpimg

path = 'dataset/horse/fake/n02391049_2_fake_B.png'
path = 'dataset/horse/real/n02381460_27.jpg'
img = mpimg.imread(path)
print(img.shape)