import cv2
import numpy as np
import os
from pathlib import Path
import random
import sys

def load_images(folder):
    """
    Load grayscale images from a specified folder.
    
    Args:
        folder (str): Path to the folder containing images.
    
    Returns:
        tuple: A list of loaded grayscale images and a list of corresponding filenames without extensions.
    """
    # List to store loaded images
    images = []
    # List to store image filenames
    filenames = []
    # Iterate through all files in the folder in sorted order
    for f in sorted(os.listdir(folder)):
        # Read the image in grayscale mode
        image = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        # Ensure the image is successfully loaded
        if (image is not None):
            # Add image to the list
            images.append(image)
            # Get the name of the file (remove the extension) and store it in the filenames list
            # .stem: returns the file name, excluding the extension
            filenames.append(Path(f).stem)
    # Return images and corresponding filenames
    return images, filenames

def detect_image_features(image):
    """
    Detect keypoints and compute descriptors using the SIFT (Scale-Invariant Feature Transform) algorithm.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
    
    Returns:
        tuple: List of keypoints and corresponding descriptors.
    """
    # Create SIFT feature detector
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors
    """
    keypoints: Stores the detected key points (feature points), a list containing cv2.KeyPoint objects.
    
    descriptors: Stores the feature descriptor of each key point, a numpy.ndarray with a 
    dimension of (n, 128), where n is the number of key points, and each key point 
    corresponds to a 128-dimensional feature vector
    """
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # Output the number of detected keypoints
    print(f"Detected {len(keypoints)} keypoints")
    return keypoints, descriptors

def feature_matching(descriptor1, descriptor2, ratio=0.75):
    """
    Perform feature matching using the ratio test with the BFMatcher.
    
    Args:
        descriptor1 (numpy.ndarray): Descriptors from the first image.
        descriptor2 (numpy.ndarray): Descriptors from the second image.
        ratio (float, optional): Lowe's ratio test threshold (default is 0.75).
    
    Returns:
        list: List of good matches that pass the ratio test.
    """
    # List to store good matches
    best_matches = []
    # Create a brute-force matcher, For SIFT and SURF, using Euclidean distance (L2 norm)
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    # Perform k-NN matching with k=2 (finding two nearest neighbors)
    closest_matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    """
    In closest_matches returned by knnMatch(), each element is a list of two DMatch objects:
    DMatch.queryIdx: index in descriptor1.
    DMatch.trainIdx: index in descriptor2.
    DMatch.distance: Euclidean distance between two feature point descriptors.
    """
    # Apply Lowe's ratio test: Only keep good matches where the distance of the best match is significantly lower than the second-best match
    for m, n in closest_matches:
        if (m.distance < ratio * n.distance):
            best_matches.append(m)
    return best_matches

def draw_matches(image1, keypoint1, image2, keypoint2, matches):
    """
    Draw matching feature points between two images with different colors.
    
    Args:
        image1 (numpy.ndarray): First input image.
        keypoint1 (list): Keypoints detected in the first image.
        image2 (numpy.ndarray): Second input image.
        keypoint2 (list): Keypoints detected in the second image.
        matches (list): List of matched keypoints.
    
    Returns:
        numpy.ndarray: Image with drawn matches.
    """
    
    """
    None: this parameter is a mask, usually set to None, indicating that all matching points are drawn.
    matchColor=(255,0,0): color of the line connecting the matching points (blue, BGR format).
    singlePointColor=(0,0,255): color of the unmatched feature points (red, BGR format).
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS: only draw matching points, do not draw unmatched points.
    """
    match_image = cv2.drawMatches(image1, keypoint1, image2, keypoint2, matches, None, 
                                  matchColor=(255,0,0), singlePointColor=(0,0,255), 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_image

def draw_epipolar_lines(image, points, F):
    """
    Draw epipolar lines on an image given a set of points and the fundamental matrix.
    
    Args:
        image (numpy.ndarray): Input image.
        points (numpy.ndarray): Points from one image to compute corresponding epipolar lines.
        F (numpy.ndarray): Fundamental matrix relating two images.
    
    Returns:
        numpy.ndarray: Image with epipolar lines drawn.
    """
    
    # Compute epipolar lines for given points in the other image
    """
    points.reshape(-1,1,2): Reshape points into the format required by OpenCV.
    1: Indicates that points comes from the first image (if it is the second image, set to 2).
    F: Basic matrix, used to calculate the epipolar equation.
    lines.reshape(-1,3): Reshape the result into an array of (N, 3), where each row (a, b, c) represents an epipolar line.
    """
    lines = cv2.computeCorrespondEpilines(points.reshape(-1,1,2), 1, F)
    lines = lines.reshape(-1,3)
    # Convert grayscale image to color for visualization
    epipolar_line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        # ax + by + c = 0
        a, b, c = line
        # Prevent division by zero
        if (abs(b) < 1e-6):
            b = 1e-6
        # Compute line start point
        left_edge_x0 = 0
        y0 = int(-c / b)
        # Compute line end point
        right_edge_x1 = image.shape[1]
        y1 = int(-(a * image.shape[1] + c) / b)
        # Random color for each line
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        # Draw the line on the image from (x0, y0) to (x1, y1).
        cv2.line(epipolar_line_image, (left_edge_x0, y0), (right_edge_x1, y1), color, 1)
    return epipolar_line_image

def blend_images(base_image, overlay_image):
    """
    Perform simple feathered blending of two images using weighted averaging.
    
    Args:
        base_image (numpy.ndarray): First input image.
        overlay_image (numpy.ndarray): Second input image.
    
    Returns:
        numpy.ndarray: Blended image.
    """
    
    # Weighted image blending allows two images to merge smoothly.
    """
    cv2.addWeighted(src1, alpha, src2, beta, gamma):
    src1: first input image (numpy.ndarray).
    alpha: weight of the first image (0.5).
    src2: second input image (numpy.ndarray).
    beta: weight of the second image (0.5).
    gamma: additional brightness adjustment (here 0, no additional adjustment)
    
    blended(x, y) = alpha * base_image(x, y) + beta * overlay_image(x, y) + gamma
    """
    blended = cv2.addWeighted(base_image, 0.5, overlay_image, 0.5, 0)
    return blended

def check_pair_and_stitch_image(image1, keypoint1, descriptor1, image2, keypoint2, descriptor2, output_dir, prefix):
    """
    Process a pair of images, performing the following steps:
    
    1. Feature matching and save match visualization.
    2. Estimate the fundamental matrix using RANSAC and visualize inlier matches and epipolar lines.
    3. Estimate the homography matrix using RANSAC and visualize inlier matches.
    4. Determine if the images can be accurately stitched.
    5. If stitchable, generate and save the stitched image.
    
    Args:
        image1 (numpy.ndarray): First input image.
        keypoint1 (list): Keypoints of the first image.
        descriptor1 (numpy.ndarray): Descriptors of the first image.
        image2 (numpy.ndarray): Second input image.
        keypoint2 (list): Keypoints of the second image.
        descriptor2 (numpy.ndarray): Descriptors of the second image.
        output_dir (str): Directory to save output images.
        prefix (str): Prefix for output filenames.
    
    Returns:
        numpy.ndarray or None: Estimated homography matrix H, or None if stitching is not possible.
    """
    # Start feature matching
    matches = feature_matching(descriptor1, descriptor2)
    if (len(descriptor1) > 0):
        match_ratio1 = len(matches) / len(descriptor1)
    else:
        match_ratio1 = 0
    if (len(descriptor2) > 0):
        match_ratio2 = len(matches) / len(descriptor2)
    else:
        match_ratio2 = 0
    print(f"Matches: {len(matches)}, fraction in image1: {match_ratio1:.2f}, fraction in image2: {match_ratio2:.2f}")
    # draw the matches line between the two images
    matches_image = draw_matches(image1, keypoint1, image2, keypoint2, matches)
    cv2.imwrite(os.path.join(output_dir, prefix + '_matches.jpg'), matches_image)
    
    # Check if images belong to the same scene
    min_matches = 10
    fraction_threshold = 0.1
    if ((len(matches) < min_matches) or (len(matches) < fraction_threshold * min(len(keypoint1), len(keypoint2)))):
        print(f"Decision: {prefix} does not belong to the same scene.")
        return None
    
    # Estimate Fundamental Matrix F
    points1 = np.float32([keypoint1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoint2[m.trainIdx].pt for m in matches])
    # Calculate the basic matrix F through RANSAC and remove outliers
    fundamental_matrix, mask_F = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 1.0, 0.99)
    
    # If F fails to be calculated, it means there are too few matching points or they are unevenly distributed.
    if (fundamental_matrix is None or mask_F is None):
        print(f"Fundamental matrix estimation failed: {prefix}")
        return None
    
    inlier_matches_F = [matches[i] for i in range(len(matches)) if mask_F[i]]
    inlier_ratio_F = 0
    if (len(matches) > 0):
        inlier_ratio_F = (len(inlier_matches_F) / len(matches)) * 100
    print(f"Inlier Ratio of Fundamental Matrix: {inlier_ratio_F:.2f}%")
    
    # Visualize an image containing only inlier matches
    inliers_image = draw_matches(image1, keypoint1, image2, keypoint2, inlier_matches_F)
    cv2.imwrite(os.path.join(output_dir, prefix + '_inlier_matches_F.jpg'), inliers_image)
    
    # Draw the epipolar lines
    # Contains matching points after RANSAC filtering, only the internal points that meet the constraints of the Fundamental Matrix are retained
    inlier_points1 = np.float32([keypoint1[m.queryIdx].pt for m in inlier_matches_F])
    inlier_points2 = np.float32([keypoint2[m.trainIdx].pt for m in inlier_matches_F])
    # make sure the points are in the right shape for cv2.computeCorrespondEpilines
    inlier_points1_reshaped = inlier_points1.reshape(-1, 1, 2)
    epilines_image2 = draw_epipolar_lines(image2, inlier_points1_reshaped, fundamental_matrix)
    cv2.imwrite(os.path.join(output_dir, prefix + '_epilines.jpg'), epilines_image2)
    
    # Estimate Homography Matrix H
    # ransacReprojThreshold = 3.0: RANSAC error threshold, in pixels. If the error of the transformed point is greater than 3.0 pixels, the point is considered an outlier.
    homography_matrix, mask_H = cv2.findHomography(inlier_points1, inlier_points2, cv2.RANSAC, 3.0)
    
    # This indicates insufficient data or poor distribution of matching points
    if (homography_matrix is None or mask_H is None):
        print(f"Homography estimation failed: {prefix}")
        return None
    
    # Statistics Homography inliers
    homography_inlier_matches = []
    for i, m in enumerate(inlier_matches_F):
        # mask_H[i] == 1 means this match is an interior point of Homography
        if (mask_H[i] == 1):
            homography_inlier_matches.append(m)
    inlier_ratio_H = 0
    if (len(inlier_matches_F) > 0):
        inlier_ratio_H = (len(homography_inlier_matches) / len(inlier_matches_F)) * 100
    print(f"Homography Inliers: {len(homography_inlier_matches)} / {len(inlier_matches_F)} ({inlier_ratio_H:.2f}%)")
    
    # Visualize matches containing only Homography inliers
    inliers_image_H = draw_matches(image1, keypoint1, image2, keypoint2, homography_inlier_matches)
    cv2.imwrite(os.path.join(output_dir, prefix + '_inlier_matches_H.jpg'), inliers_image_H)
    
    # Make a final decision based on the inlier relationship between F and H
    # If most (e.g. >= 50%) of the inliers in F are still inliers in H, it is determined that the alignment can be done accurately
    THRESHOLD_RATIO = 50.0
    if (inlier_ratio_H >= THRESHOLD_RATIO):
        print(f"Decision: YES, {prefix} can be accurately aligned.")
        print(f"Reason: {inlier_ratio_H:.2f}% of F inliers also passed the homography test.")
    else:
        print(f"Decision: NO, {prefix} cannot be accurately aligned.")
        print(f"Reason: Only {inlier_ratio_H:.2f}% of F inliers remained in H inliers, which is below the threshold.")
        return None
    
    # Stitch Images
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    canvas_width = width1 + width2
    canvas_height = max(height1, height2)
    """
    cv2.warpPerspective(src, H, dsize) uses H to transform image1 to align image2.
    (canvas_width, canvas_height) specifies the size of the transformed image so that the transformed image1 can be fully displayed on the new canvas.
    """
    warped_image1 = cv2.warpPerspective(image1, homography_matrix, (canvas_width, canvas_height))
    mosaic = warped_image1.copy()
    roi = mosaic[0:height2, 0:width2]
    mosaic[0:height2, 0:width2] = blend_images(roi, image2)
    cv2.imwrite(os.path.join(output_dir, prefix + '_mosaic.jpg'), mosaic)
    return homography_matrix

def find_connected_components(graph, n):
    """
    Find connected components in an undirected graph.
    
    Args:
        graph (dict): A dictionary where keys are nodes, and values are sets of adjacent nodes.
        n (int): The total number of nodes in the graph.
    
    Returns:
        list: A list of connected components, where each component is represented as a list of nodes.
    """
    # Initialize a visited list to track which nodes have been explored
    visited = [False] * n
     # List to store all connected components
    components = []

    # Iterate through each node to find unvisited components
    for i in range(n):
        # If node i is not visited, start a new component search
        if (visited[i] == False):
            # List to store the current connected component
            current_connected_component_nodes = []
            # Use a queue for BFS traversal
            queue = [i]
            # Mark the node as visited
            visited[i] = True
            # Perform BFS to explore the component
            while (queue):
                # Get the current node
                current_node = queue.pop(0)
                # Add it to the current component
                current_connected_component_nodes.append(current_node)
                # Iterate through its neighbors
                # graph.get(cur, []) avoids KeyError and ensures that cur is not in graph and can be executed safely
                for neighbor in graph.get(current_node, []):
                    # If neighbor is unvisited, mark and enqueue it
                    if (visited[neighbor] == False):
                        visited[neighbor] = True
                        queue.append(neighbor)
            # Add the found component to the list
            components.append(current_connected_component_nodes)
    return components

def multi_image_mosaics(images, components, graph_h, output_dir, filenames):
    """
    Construct a multi-image mosaic using a connected component in the graph.
    
    Args:
        images (list): List of input images.
        components (list): List of indices representing the connected component of images.
        graph_h (dict): Dictionary storing homography matrices between image pairs, with keys as (i, j).
        output_dir (str): Directory to save the output mosaic.
        filenames (list): List of filenames corresponding to the images.
    
    Returns:
        None. The resulting mosaic is saved as an image file.
    """
    
    # Sort the component indices to ensure a consistent processing order
    comp_sorted = sorted(components)
    
    # Select the anchor image (middle image in the sorted component)
    anchor = comp_sorted[len(comp_sorted) // 2]
    print(f"Multi-image stitching: Chosen anchor image {filenames[anchor]}")
    
    # Compute transformation matrices for all images relative to the anchor using BFS
    # Identity matrix for the anchor image
    transformations = {anchor: np.eye(3)}
    visited = set([anchor])
    queue = [anchor]
    
    # Construct adjacency list for the undirected graph
    adj = {i: [] for i in components}
    for (i, j) in graph_h.keys():
        if ((i in components) and (j in components)):
            adj[i].append(j)
            adj[j].append(i)
    
    # BFS to compute transformations for each image relative to the anchor
    while queue:
        current = queue.pop(0)
        for neighbor in adj[current]:
            if neighbor not in visited:
                # Use the direct homography if available, otherwise use the inverse of the reverse homography
                if ((neighbor, current) in graph_h):
                    H_nc = graph_h[(neighbor, current)]
                elif ((current, neighbor) in graph_h):
                    H_nc = np.linalg.inv(graph_h[(current, neighbor)])
                else:
                    continue
                # Compute transformation relative to the anchor
                transformations[neighbor] = transformations[current] @ H_nc
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Determine the size of the output canvas by transforming image corners
    corners = []
    for i in components:
        # Get image dimensions
        height, width = images[i].shape
        # Image corners
        points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
        # Transform corners
        points_trans = cv2.perspectiveTransform(points, transformations[i])
        corners.extend(points_trans.reshape(-1, 2))
    
    corners = np.array(corners)
    min_x, min_y = corners.min(axis=0) # Find the minimum x and y coordinates
    max_x, max_y = corners.max(axis=0) # Find the maximum x and y coordinates
    canvas_width = int(np.ceil(max_x - min_x)) # Determine canvas width
    canvas_height = int(np.ceil(max_y - min_y)) # Determine canvas height
    
    # Construct a translation matrix to shift the entire mosaic to positive coordinates
    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]])
    
    # Initialize the output mosaic and weight matrices
    mosaic = np.zeros((canvas_height, canvas_width), dtype=np.float32) # Initialize mosaic canvas
    weight = np.zeros((canvas_height, canvas_width), dtype=np.float32) # Initialize weight matrix
    
    # Warp each image into the canvas and blend them together
    for i in components:
        # Apply the total transformation including the translation
        H_total = T @ transformations[i]
        # Warp the image
        warped = cv2.warpPerspective(images[i], H_total, (canvas_width, canvas_height))
        
        # Create a binary mask where valid pixels exist
        binary_mask = (warped > 0).astype(np.float32)
        # Accumulate the pixel values
        mosaic += warped.astype(np.float32) * binary_mask
        # Accumulate the mask values to normalize the blending
        weight += binary_mask
    
    # Avoid division by zero in the weight matrix
    weight[weight == 0] = 1
    # Normalize pixel values and convert to uint8
    mosaic_final = (mosaic / weight).astype(np.uint8)
    
    # Save the final mosaic image
    mosaic_filename = os.path.join(output_dir, '_'.join(sorted([filenames[i] for i in components])) + '_multi_mosaic.jpg')
    cv2.imwrite(mosaic_filename, mosaic_final)
    print(f"Saved multi-image mosaic: {mosaic_filename}")

def image_mosaic_running(input_dir, output_dir):
    """
    Main function to process image pairs, detect features, match them, and build a mosaic for the largest connected component.

    Args:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the output results.
    
    Returns:
        None. The function processes image pairs, stores homography matrices, and attempoints to build a multi-image mosaic.
    """

    # Load images and their corresponding filenames
    images, filenames = load_images(input_dir)
    # Number of images
    n = len(images)

    # Detect keypoints and descriptors for each image
    # List to store keypoints for each image
    all_keypoints = []
    # List to store descriptors for each image
    all_descriptors = []

    for image in images:
        # Extract SIFT keypoints and descriptors
        keypoint, descriptor = detect_image_features(image)
        all_keypoints.append(keypoint)
        all_descriptors.append(descriptor)

    # Store successfully matched image pairs and build a connectivity graph
    # Dictionary to store homography matrices, keys are (i, j)
    graph_h = {}
    # Connectivity graph for image pairs
    # connectivity = {i: set() for i in range(n)}
    connectivity = {}
    for i in range(n):
        connectivity[i] = set()
    # Process every pair of images
    for i in range(n):
        # Only consider unique pairs (i, j)
        for j in range(i + 1, n):
            # Generate output prefix
            prefix = '_'.join(sorted([filenames[i], filenames[j]]))
            print(f"\nProcessing image pair: {filenames[i]} and {filenames[j]}")

            # Compute homography matrix between image i and image j
            H = check_pair_and_stitch_image(images[i], all_keypoints[i], all_descriptors[i], images[j], all_keypoints[j], all_descriptors[j], output_dir, prefix)

            if (H is not None):
                # Store homography matrix for (i, j)
                graph_h[(i, j)] = H

                # Attempt to store the inverse homography matrix for (j, i)
                try:
                    # Calculate the inverse matrix H_inv of H and store it in graph_h[(j, i)].
                    # H_inv is used to transform from image_j back to image_i
                    H_inv = np.linalg.inv(H)
                    graph_h[(j, i)] = H_inv
                except np.linalg.LinAlgError:
                    pass  # If the matrix is singular, skip storing its inverse

                # Update the connectivity graph
                connectivity[i].add(j)
                connectivity[j].add(i)
                
    # Find the largest connected component of images
    components = find_connected_components(connectivity, n)
    if (components):
        # Get the largest connected component
        largest_comp = max(components, key=lambda x: len(x))
        if (len(largest_comp) > 1):
            print(f"\nLargest connected component contains {len(largest_comp)} images.")
            # Attempt to construct a multi-image mosaic
            multi_image_mosaics(images, largest_comp, graph_h, output_dir, filenames)
        else:
            print("Not enough images for multi-image stitching.")
    else:
        print("No connected components found.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python job1_align.py in_dir out_dir")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # Create output directory if it does not exist
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    image_mosaic_running(input_dir, output_dir)