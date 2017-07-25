# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[car]: ./writeup_images/car_image.png
[notcar]: ./writeup_images/notcar_image.png
[hog]: ./writeup_images/hog_features.png
[slide]: ./writeup_images/sliding_window.png
[heatmaps]: ./writeup_images/heatmaps.png
[combined]: ./writeup_images/combined_bboxes.png
[vidout]: ./writeup_images/vidout.png
[video1]: ./finalt_video.mp4

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. You can use this template writeup for this project as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells 1 through 5 of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car] ![alt text][notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` over the car and not car images displayed before:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, like different color spaces orietations and some others. I settled to the ones I could really get a feel of what the image was like looking through the hog features only.

The HOG parameters chosen were:

```python
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, 3, or "ALL"

```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

On the 6th code cellm, I trained a linear SVM classifier using spatial, histogram,  and HOG features. I also implemented a convertion to `YCrCb` color space. Besides the HOG parameters stated earlier, the other parameters were set to.
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32) 
hist_bins = 32 
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features features on or off
```
These where a bit trickier to chose since the changes weren't very noticeable depending on the magnitude of the change. But following some advices from the forums and in the lesson I ended up with this set

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to apply the window search only over the pixels between `y=(400,656)` since I shouldn't expect to find vehicles in the top half of the image. 50% of overlap and a scaled of 48 identified well the parts of the car, and later showed good results. Here are some examples

![alt text][slide]

That was done on the 7th code cell.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on one a half scales using YCrCb ALL-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][heatmaps]

Noticing there where some false positives in the test images, I applied a heatmap threshold of 1, that even being very small, erased the false positives from the images.

I also used the `scipy.ndimage.measurements.label()` to combine the multiple detections that can be clearly seen in the last image. Here is the result of this combination:

![alt text][combined]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./final_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I also applied the same threshold I tested on the images to the video. The pipeline with the threshold performed much better and there were nearly no false positives. The `apply_threshold()` is defined on the 12th code cell.

As suggested on the review, I stored the heatmaps on a `deque` of `max_len` 10 and averaged them to smooth the drawing boxes for every 10 frames. That was done in the 11th code cell and implemented on the 12th as well.

I then generated another video with the boxes and the heatmap side by side.

![alt_text][vidout]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One thing that my pipeline is not able to do is identify different cars in an image. If I had implemented a class to average the position of the cars found I would probably be able to do that. One time during the video, bounding boxes from different cars merged together. I saw a video from another student where he managed to do that.

This approach would help even more to prevent false positives.

---

This is the edited version following the [revisions](https://review.udacity.com/#!/reviews/630509). The first version can be found [here](https://github.com/lealldiogo/CarND-Vehicle-Detection)
