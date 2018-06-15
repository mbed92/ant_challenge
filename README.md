# Challenge from Tooplox
I have images of an ant in the *bug* folder. The task is to come up with an algorithm to combine all images to produce a single, sharp color image and also a grayscale image of an estimated depth map (white - nearest, black - farthest). Use OpeCV... but the tricky part comes here - use it only to read and wirite :|

# Howto
* mkdir build && cd build
* cmake ..
* make
* ./TooploxBlurDeph <list_of_files> <output_rgb_image.png> <output_depth_map.png>

# Also
Play with generated images using http://depthy.me 

# Results
![alt text](https://raw.githubusercontent.com/mbed92/ant_challenge/deblurred.png)
![alt text](https://raw.githubusercontent.com/mbed92/ant_challenge/depth.png)
