
% Accurate nuclei identification in pancreas tissue sample scan

clear variables;
close all;


%% pixel color classifier
% Step 1: Load the training slide and ask the user to select a region that
% contains nucleus pixels.
training_slide = im2double(imread('pathology_slide_1.png'));

training_slide_fig = figure;
imshow(training_slide);
title('Select a region consisting of nucleus pixels. Double-click to confirm.');
imrect_handle = imrect;
nucleus_rect = round(wait(imrect_handle));

nucleus_region = training_slide(nucleus_rect(2) : nucleus_rect(2) + nucleus_rect(4), ...
    nucleus_rect(1) : nucleus_rect(1) + nucleus_rect(3), :);

figure; imshow(nucleus_region);
title('Nucleus Region of Interest');

set(0, 'CurrentFigure', training_slide_fig);
title('Select a region consisting of non-nucleus pixels. Double-click to confirm.');
imrect_handle = imrect;
bg_rect = round(wait(imrect_handle));

bg_region = training_slide(bg_rect(2) : bg_rect(2) + bg_rect(4), ...
    bg_rect(1) : bg_rect(1) + bg_rect(3), :);

figure; imshow(bg_region);
title('Non-Nucleus Region of Interest');

% Step 2: Compute RGB, HSV, and LAB color feature vector for the nucleus
% and the background pixels.
nucleus_rgb = reshape(nucleus_region, [], 3);
nucleus_hsv = rgb2hsv(nucleus_rgb);
nucleus_lab = rgb2lab(nucleus_rgb);
nucleus_feature_matrix = [nucleus_rgb nucleus_hsv nucleus_lab];

bg_rgb = reshape(bg_region, [], 3);
bg_hsv = rgb2hsv(bg_rgb);
bg_lab = rgb2lab(bg_rgb);
bg_feature_matrix = [bg_rgb bg_hsv bg_lab];

% Step 3: Create a linear regression classifier based on colors.
feature_matrix = [nucleus_feature_matrix; bg_feature_matrix];
ground_truth_vector = [ones(size(nucleus_feature_matrix, 1), 1); ...
    zeros(size(bg_feature_matrix, 1), 1)];
linear_regression_model = fitlm(feature_matrix, ground_truth_vector);

% Step 4: Create a binary map using the linear regression classifier.
training_slide_rgb = reshape(training_slide, [], 3);
training_slide_hsv = rgb2hsv(training_slide_rgb);
training_slide_lab = rgb2lab(training_slide_rgb);
training_slide_feature_matrix = [training_slide_rgb training_slide_hsv ...
    training_slide_lab];

prediction_vector = predict(linear_regression_model, ...
    training_slide_feature_matrix);
prediction_map = reshape(prediction_vector, size(training_slide, 1), ...
    size(training_slide, 2));

figure; imshow(prediction_map, []);
title('Nucleus Prediction Map');

%% morphological detection

% Step 1: Apply a Gaussian filter to the nucleus map.
training_slide = im2double(imread('pathology_slide_1.png'));
nucleus_map = im2double(imread('nucleus_prediction_map.png'));
nucleus_map_smoothed = imgaussfilt(nucleus_map, 2);

figure; imshow(nucleus_map);
figure; imshow(nucleus_map_smoothed);

% Step 2: Apply a morphological opening to detect circular blobs.
se = strel('disk', 10);
nucleus_map_blobs = imopen(nucleus_map_smoothed, se);

figure; imshow(nucleus_map_blobs);

% Step 3: Find regional maxima that meet a threshold.
cell_markers = imregionalmax(nucleus_map_blobs);
cell_markers_cc = bwconncomp(cell_markers);

figure; imshow(cell_markers);

% Step 4: Compute the seed points.
cell_markers_cc = bwconncomp(cell_markers);
cell_markers_stats = regionprops(cell_markers_cc, 'Centroid');
seed_map = zeros(size(nucleus_map));

for i = 1 : cell_markers_cc.NumObjects
    centroid = round(cell_markers_stats(i).Centroid);
    seed_map(centroid(2), centroid(1)) = 1;
end

seed_map = seed_map > 0;

figure; imshow(imdilate(seed_map, strel('disk', 3)));
figure; imshow(imfuse(training_slide, ...
    imdilate(seed_map, strel('disk', 3))));

%% watershed segmentation

nucleus_map = im2double(imread('nucleus_prediction_map.png'));
nucleus_map_smoothed = imgaussfilt(nucleus_map, 5);

se = strel('disk', 8);
se2 = strel(ones(3,3));

% Step 1: Apply an opening by reconstruction.
nucleus_map_e = imerode(nucleus_map_smoothed, se);
nucleus_map_obr = imreconstruct(nucleus_map_e, nucleus_map_smoothed);

figure; imshow(nucleus_map_obr, []);

% Step 2: Apply a closing by reconstruction.
nucleus_map_obrd = imdilate(nucleus_map_obr, se);
nucleus_map_obrcbr = imreconstruct(imcomplement(nucleus_map_obrd), ...
    imcomplement(nucleus_map_obr));
nucleus_map_obrcbr = imcomplement(nucleus_map_obrcbr);

figure; imshow(nucleus_map_obrcbr, []);

% Step 3: Find the regional maxima.
regional_max = imregionalmax(nucleus_map_obrcbr);
figure; imshow(regional_max, []);

nucleus_map_max = nucleus_map_smoothed;
nucleus_map_max(regional_max) = 1;
figure; imshow(nucleus_map_max, []);

regional_max = imclose(regional_max, se2);
regional_max = imerode(regional_max, se2);
regional_max = bwareaopen(regional_max, 10);
nucleus_map_max = nucleus_map_smoothed;
nucleus_map_max(regional_max) = 1;

figure; imshow(nucleus_map_max, []);

% Step 4: Identify the watershed ridge lines.
cell_markers = imbinarize(nucleus_map_obrcbr);
cm_distance_map = -bwdist(~cell_markers);
cm_watershed = watershed(cm_distance_map);
cm_watershed_ridge_lines = (cm_watershed == 0);

figure; imshow(cell_markers, []);
figure; imshow(cm_distance_map, []);
figure; imshow(cm_watershed, []);
figure; imshow(imfuse(cell_markers, cm_watershed_ridge_lines), []);

% Step 6: Perform watershed segmentation.
nucleus_map_gradient = imgradient(nucleus_map_smoothed);
watershed_input_map = imimposemin(nucleus_map_gradient, ...
    regional_max | cm_watershed_ridge_lines);
nucleus_segmentation = watershed(watershed_input_map) > 1;

% Step 7: Remove small connected components.
nucleus_segmentation_cc = bwconncomp(nucleus_segmentation);

comp_sizes = cellfun(@numel, nucleus_segmentation_cc.PixelIdxList);
figure; histogram(comp_sizes);
title('Size Histogram');

for i = 1 : nucleus_segmentation_cc.NumObjects
    if numel(nucleus_segmentation_cc.PixelIdxList{i}) <= 50
        nucleus_segmentation(nucleus_segmentation_cc.PixelIdxList{i}) = 0;
    end
end

figure; imshow(nucleus_map_gradient, []);
figure; imshow(imfuse(cm_watershed_ridge_lines, nucleus_map_gradient), []);
figure; imshow(watershed_input_map, []);
figure; imshow(nucleus_segmentation, []);

figure; imshow(imfuse(nucleus_map_gradient, regional_max));
figure; imshow(imfuse(nucleus_map, nucleus_segmentation));


