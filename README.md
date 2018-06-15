# Challenge from Tooplox
I have images of an ant in the *bug* folder. The task is to come up with an algorithm to combine all images to produce a single, sharp color image and also a grayscale image of an estimated depth map (white - nearest, black - farthest). Use OpeCV... but the tricky part comes here - use it only to read and wirite :|

# Howto
* mkdir build && cd build
* cmake ..
* make
* ./TooploxBlurDepth <list_of_files> <output_rgb_image.png> <output_depth_map.png>

# Also
Play with generated images using http://depthy.me 

# Tested on
* OpenCV 3.3.0
* CMake 2.8
* C++14

# Results
![](https://github.com/mbed92/ant_challenge/blob/master/depth.png)
![](https://github.com/mbed92/ant_challenge/blob/master/deblurred.png)
