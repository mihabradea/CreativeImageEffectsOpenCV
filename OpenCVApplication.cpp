// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <cstdlib>


wchar_t* projectPath;


void applySepiaEffect(Mat& src, Mat& dst) {
	// Check if the image is grayscale
	if (src.channels() == 1) {
		cvtColor(src, src, COLOR_GRAY2BGR); // Convert grayscale to BGR
	}
	dst = src.clone();
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			Vec3b pixel = src.at<Vec3b>(y, x);
			// Original RGB values
			float blue = pixel[0];
			float green = pixel[1];
			float red = pixel[2];
			// Calculate new RGB values
			float newRed = (red * 0.393) + (green * 0.769) + (blue * 0.189);
			float newGreen = (red * 0.349) + (green * 0.686) + (blue * 0.168);
			float newBlue = (red * 0.272) + (green * 0.534) + (blue * 0.131);
			// Set new RGB values
			dst.at<Vec3b>(y, x)[2] = min((int)newRed, 255);
			dst.at<Vec3b>(y, x)[1] = min((int)newGreen, 255);
			dst.at<Vec3b>(y, x)[0] = min((int)newBlue, 255);
		}
	}
}

void testSepiaEffect() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty()) {
			printf("Failed to read image: %s\n", fname);
			return;
		}
		Mat dst;
		applySepiaEffect(src, dst);
		imshow("Original Image", src);
		imshow("Sepia Effect", dst);
		waitKey(0);
	}
}


void applyRainbowEffect(Mat& src, Mat& dst) {
	dst = src.clone();
	// Define the rainbow colors
	Scalar colors[] = {
		Scalar(255, 0, 0),   // Blue
		Scalar(0, 255, 0),   // Green
		Scalar(0, 0, 255),   // Red
		Scalar(255, 255, 0), // Cyan
		Scalar(255, 0, 255), // Magenta
		Scalar(0, 255, 255)  // Yellow
	};
	int numColors = sizeof(colors) / sizeof(colors[0]);
	// Calculate the width of each color band
	int bandWidth = src.cols / numColors;
	int lastBandWidth = src.cols - (bandWidth * (numColors - 1)); // This ensures full coverage of the image width
	for (int i = 0; i < numColors; ++i) {
		// Define the start and end columns for the current color band
		int startX = i * bandWidth;
		int endX = (i == numColors - 1) ? src.cols : (startX + bandWidth); // Adjust the last band to cover the rest
		// Blend the current color with the image
		for (int y = 0; y < src.rows; ++y) {
			for (int x = startX; x < endX; ++x) {
				// Access the pixel in the original image
				Vec3b& pixel = dst.at<Vec3b>(y, x);
				// Access the RGB components of the current color
				uchar blue = colors[i][0];
				uchar green = colors[i][1];
				uchar red = colors[i][2];
				// Blend the rainbow color with the original pixel
				pixel[0] = saturate_cast<uchar>((pixel[0] + blue) / 2); // Blue channel
				pixel[1] = saturate_cast<uchar>((pixel[1] + green) / 2); // Green channel
				pixel[2] = saturate_cast<uchar>((pixel[2] + red) / 2); // Red channel
			}
		}
	}
}

void testRainbowEffect() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty()) {
			printf("Failed to read image: %s\n", fname);
			return;
		}
		Mat rainbowImage;
		applyRainbowEffect(src, rainbowImage);
		imshow("Original Image", src);
		imshow("Rainbow Effect", rainbowImage);
		waitKey(0);
	}
}


void applyMosaicEffect(Mat& src, Mat& dst, int blockSize)
{
	dst = src.clone(); 
	// Loop through the image in steps of blockSize
	for (int y = 0; y < src.rows; y += blockSize)
	{
		for (int x = 0; x < src.cols; x += blockSize)
		{
			// Calculate the region of interest for the current block
			int blockWidth = blockSize;
			int blockHeight = blockSize;
			if (x + blockSize > src.cols)
				blockWidth = src.cols - x;
			if (y + blockSize > src.rows)
				blockHeight = src.rows - y;
			// Calculate the average color of the block
			Scalar avgColor = mean(src(Rect(x, y, blockWidth, blockHeight)));
			// Fill the block with the average color
			rectangle(dst, Rect(x, y, blockWidth, blockHeight), avgColor, FILLED);
		}
	}
}

void testMosaicEffect()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		if (src.empty())
		{
			printf("Failed to read image: %s\n", fname);
			continue;
		}
		int blockSize;
		printf("Enter block size for mosaic effect: ");
		scanf("%d", &blockSize);
		Mat dst;
		applyMosaicEffect(src, dst, blockSize);
		imshow("Original Image", src);
		imshow("Mosaic Effect", dst);
		waitKey(0);
	}
}

void applyOilPaintEffect(Mat& src, Mat& dst, int radius, int levels)
{
	dst = src.clone(); 
	// Loop through the image in blocks
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			// Get the region of interest (ROI) for the current block
			int startY = y - radius;
			int startX = x - radius;
			int endY = y + radius + 1;
			int endX = x + radius + 1;
			// Ensure ROI boundaries are within image dimensions
			if (startY < 0) startY = 0;
			if (startX < 0) startX = 0;
			if (endY > src.rows) endY = src.rows;
			if (endX > src.cols) endX = src.cols;
			// Initialize histograms for each channel
			std::vector<int> histR(levels, 0);
			std::vector<int> histG(levels, 0);
			std::vector<int> histB(levels, 0);
			// Calculate histograms within the block
			for (int i = startY; i < endY; i++)
			{
				for (int j = startX; j < endX; j++)
				{
					Vec3b intensity = src.at<Vec3b>(i, j);
					histR[intensity[2] * levels / 256]++;
					histG[intensity[1] * levels / 256]++;
					histB[intensity[0] * levels / 256]++;
				}
			}
			// Find the most frequent color within each channel
			int maxIndexR = 0, maxIndexG = 0, maxIndexB = 0;
			for (int i = 1; i < levels; i++)
			{
				if (histR[i] > histR[maxIndexR])
					maxIndexR = i;
				if (histG[i] > histG[maxIndexG])
					maxIndexG = i;
				if (histB[i] > histB[maxIndexB])
					maxIndexB = i;
			}
			// Set the color of the block to the most frequent color
			Vec3b& pixel = dst.at<Vec3b>(y, x);
			pixel[2] = maxIndexR * 256 / levels;
			pixel[1] = maxIndexG * 256 / levels;
			pixel[0] = maxIndexB * 256 / levels;
		}
	}
}

void testOilPaintEffect()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); 
		Mat src = imread(fname);
		if (src.empty())
		{
			printf("Failed to read image: %s\n", fname);
			continue;
		}
		int radius, levels;
		printf("Enter radius for oil painting effect: ");
		scanf("%d", &radius);
		printf("Enter number of intensity levels: ");
		scanf("%d", &levels);
		Mat dst;
		applyOilPaintEffect(src, dst, radius, levels);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("input image", src);
		imshow("oil painting effect", dst);
		waitKey();
	}
}

void applyVignette(Mat& src, Mat& dst)
{
	Mat mask = Mat::zeros(src.size(), src.type());
	Point center = Point(mask.cols / 2, mask.rows / 2);
	double maxDistance = sqrt(center.x * center.x + center.y * center.y);
	for (int y = 0; y < mask.rows; ++y)
	{
		for (int x = 0; x < mask.cols; ++x)
		{
			double distance = sqrt((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y));
			double vignetteFactor = 1 - (distance / maxDistance);
			vignetteFactor = pow(vignetteFactor, 2.5);
			mask.at<Vec3b>(y, x) = Vec3b(vignetteFactor * 255, vignetteFactor * 255, vignetteFactor * 255);
		}
	}
	Mat temp;
	src.convertTo(temp, CV_32F);
	mask.convertTo(mask, CV_32F, 1.0 / 255.0);
	multiply(temp, mask, temp);
	temp.convertTo(dst, CV_8UC3);
}

void applyBlur(Mat& src, Mat& dst)
{
	GaussianBlur(src, dst, Size(7, 7), 0);
}

void applyRetricaEffect(Mat& src, Mat& dst)
{
	Mat sepiaImg, vignetteImg, blurredImg;
	// Apply sepia effect
	applySepiaEffect(src, sepiaImg);
	// Apply vignette effect
	applyVignette(sepiaImg, vignetteImg);
	// Apply slight blur
	applyBlur(vignetteImg, dst);
}

void testRetricaEffect()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		if (src.empty())
		{
			printf("Failed to read image: %s\n", fname);
			continue;
		}
		Mat dst;
		applyRetricaEffect(src, dst);
		imshow("Original Image", src);
		imshow("Retrica Effect", dst);
		waitKey(0);
	}
}

void apply3DAnaglyphEffect(Mat& src, Mat& dst)
{
	// Convert to grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	// Create the red and cyan images
	Mat redChannel = Mat::zeros(src.size(), CV_8UC3);
	Mat cyanChannel = Mat::zeros(src.size(), CV_8UC3);
	// Assign the grayscale image to the red channel of the red image
	std::vector<Mat> redChannels = { Mat::zeros(src.size(), CV_8UC1), Mat::zeros(src.size(), CV_8UC1), gray };
	merge(redChannels, redChannel);
	// Assign the grayscale image to the green and blue channels of the cyan image
	std::vector<Mat> cyanChannels = { gray, gray, Mat::zeros(src.size(), CV_8UC1) };
	merge(cyanChannels, cyanChannel);
	// Shift the cyan image to the right
	int shift = 10;
	Mat shiftedCyan = Mat::zeros(src.size(), CV_8UC3);
	cyanChannel(Rect(shift, 0, cyanChannel.cols - shift, cyanChannel.rows)).copyTo(shiftedCyan(Rect(0, 0, cyanChannel.cols - shift, cyanChannel.rows)));
	// Combine the red and cyan images to create the anaglyph
	addWeighted(redChannel, 0.5, shiftedCyan, 0.5, 0, dst);
}

void test3DAnaglyphEffect()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		if (src.empty())
		{
			printf("Failed to read image: %s\n", fname);
			continue;
		}
		Mat dst;
		apply3DAnaglyphEffect(src, dst);
		imshow("Original Image", src);
		imshow("3D Anaglyph Effect", dst);
		waitKey(0);
	}
}

int customMin(int a, int b) {
	return a < b ? a : b;
}

int customMax(int a, int b) {
	return a > b ? a : b;
}

void createCrossWavesPattern(Mat& src, Mat& dst, float amplitudeX, float frequencyX, float amplitudeY, float frequencyY)
{
	dst = src.clone();
	int height = src.rows;
	int width = src.cols;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int newX = x + static_cast<int>(amplitudeX * sin(2 * CV_PI * y * frequencyX / height));
			int newY = y + static_cast<int>(amplitudeY * sin(2 * CV_PI * x * frequencyY / width));
			// Ensure the new coordinates are within image bounds
			newX = customMin(customMax(newX, 0), width - 1);
			newY = customMin(customMax(newY, 0), height - 1);
			dst.at<Vec3b>(newY, newX) = src.at<Vec3b>(y, x);
		}
	}
}

void testCrossWavesPattern()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Start time measurement
		Mat src = imread(fname);
		if (src.empty())
		{
			printf("Failed to read image: %s\n", fname);
			continue;
		}
		Mat dst;
		float amplitudeX, frequencyX, amplitudeY, frequencyY;
		printf("Enter horizontal wave amplitude: ");
		scanf("%f", &amplitudeX);
		printf("Enter horizontal wave frequency: ");
		scanf("%f", &frequencyX);
		printf("Enter vertical wave amplitude: ");
		scanf("%f", &amplitudeY);
		printf("Enter vertical wave frequency: ");
		scanf("%f", &frequencyY);
		createCrossWavesPattern(src, dst, amplitudeX, frequencyX, amplitudeY, frequencyY);
		imshow("input image", src);
		imshow("cross waves pattern", dst);
		waitKey();
	}
}

void applyPencilSketchEffect(Mat& src, Mat& dst) {
	// Convert to grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	// Invert the grayscale image
	Mat grayInv;
	bitwise_not(gray, grayInv);
	// Apply Gaussian Blur
	Mat blurred;
	GaussianBlur(grayInv, blurred, Size(21, 21), 0, 0);
	// Blend the grayscale image with the blurred inverted image
	Mat sketch;
	divide(gray, 255 - blurred, sketch, 256.0);
	// Enhance the sketch by increasing the contrast
	normalize(sketch, dst, 0, 255, NORM_MINMAX);
}

void testPencilSketchEffect() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty()) {
			printf("Failed to read image: %s\n", fname);
			return;
		}
		Mat dst;
		applyPencilSketchEffect(src, dst);
		imshow("Original Image", src);
		imshow("Pencil Sketch Effect", dst);
		waitKey(0);
	}
}

void applyPsychedelicGlitchEffect(Mat& src, Mat& dst) {
	srand(time(0));  // Seed random number generator
	dst = src.clone();

	// Apply random color channel shifts
	int shift_amount = 5 + rand() % 20;  // Random shift between 5 and 25 pixels
	for (int y = 0; y < dst.rows; y++) {
		for (int x = 0; x < dst.cols; x++) {
			int shift_x = (x + shift_amount) % dst.cols;
			int shift_y = (y + shift_amount) % dst.rows;

			Vec3b color = dst.at<Vec3b>(y, x);
			Vec3b shifted_color = dst.at<Vec3b>(shift_y, shift_x);

			// Swap channels
			int channel = rand() % 3;
			color[channel] = shifted_color[channel];

			dst.at<Vec3b>(y, x) = color;
		}
	}

	// Add horizontal and vertical scan lines
	int line_frequency = 5 + rand() % 10;  // Frequency of lines
	for (int y = 0; y < dst.rows; y += line_frequency) {
		for (int x = 0; x < dst.cols; x++) {
			dst.at<Vec3b>(y, x) = Vec3b(0, 0, 0);  // Make the line black
		}
	}

	for (int x = 0; x < dst.cols; x += line_frequency) {
		for (int y = 0; y < dst.rows; y++) {
			dst.at<Vec3b>(y, x) = Vec3b(0, 0, 0);  // Make the line black
		}
	}

	// Introduce random noise
	for (int y = 0; y < dst.rows; y++) {
		for (int x = 0; x < dst.cols; x++) {
			if (rand() % 100 < 10) {  // 10% chance of noise
				Vec3b& color = dst.at<Vec3b>(y, x);
				color[0] = rand() % 256;
				color[1] = rand() % 256;
				color[2] = rand() % 256;
			}
		}
	}
}

void testPsychedelicGlitchEffect() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty()) {
			printf("Failed to read image: %s\n", fname);
			return;
		}
		Mat dst;
		applyPsychedelicGlitchEffect(src, dst);
		imshow("Original Image", src);
		imshow("Psychedelic Glitch Effect", dst);
		waitKey(0);
	}
}

void applyFrostedGlassEffect(Mat& src, Mat& dst, int blockSize) {
	dst = src.clone();  // Copy the source image to the destination

	srand(static_cast<unsigned int>(time(0)));  // Seed for randomness

	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			int randomX = x + (rand() % (2 * blockSize + 1)) - blockSize;
			int randomY = y + (rand() % (2 * blockSize + 1)) - blockSize;

			// Ensure the new coordinates are within image bounds
			randomX = customMax(0, customMin(randomX, src.cols - 1));
			randomY = customMax(0, customMin(randomY, src.rows - 1));

			dst.at<Vec3b>(y, x) = src.at<Vec3b>(randomY, randomX);
		}
	}
}

void testFrostedGlassEffect() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty()) {
			printf("Failed to read image: %s\n", fname);
			return;
		}
		Mat dst;
		int blockSize = 5;  // You can adjust blockSize to get different levels of blurriness
		applyFrostedGlassEffect(src, dst, blockSize);
		imshow("Original Image", src);
		imshow("Frosted Glass Effect", dst);
		waitKey(0);
	}
}

void applyNegativeWithRedEdgesEffect(Mat& src, Mat& dst) {
	// Convert the source image to a negative image
	Mat negative = Mat::zeros(src.size(), src.type());
	bitwise_not(src, negative);
	// Convert to grayscale for edge detection
	Mat gray;
	cvtColor(negative, gray, COLOR_BGR2GRAY);
	// Edge detection using the Canny algorithm
	Mat edges;
	Canny(gray, edges, 100, 200);  // Adjust these thresholds for sensitivity
	// Convert detected edges to BGR format
	cvtColor(edges, edges, COLOR_GRAY2BGR);
	// Prepare to highlight edges in red
	for (int y = 0; y < edges.rows; y++) {
		for (int x = 0; x < edges.cols; x++) {
			Vec3b color = edges.at<Vec3b>(y, x);
			// Wherever the edges are detected (white pixels), set to red
			if (color[0] == 255 && color[1] == 255 && color[2] == 255) {
				negative.at<Vec3b>(y, x)[0] = 0;   // Blue channel
				negative.at<Vec3b>(y, x)[1] = 0;   // Green channel
				negative.at<Vec3b>(y, x)[2] = 255; // Red channel
			}
		}
	}
	dst = negative.clone();
}

void testNegativeWithRedEdgesEffect() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname);
		if (src.empty()) {
			printf("Failed to read image: %s\n", fname);
			return;
		}
		Mat dst;
		applyNegativeWithRedEdgesEffect(src, dst);
		imshow("Original Image", src);
		imshow("Negative with Red Edges Effect", dst);
		waitKey(0);
	}
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Sepia effect\n");
		printf(" 2 - Rainbow effect\n");
		printf(" 3 - Mosaic effect\n");
		printf(" 4 - Oil painting effect\n");
		printf(" 5 - Drama effect with vignette\n");
		printf(" 6 - 3D glass anaglyph effect\n");
		printf(" 7 - Crosswaves pattern\n");
		printf(" 8 - Pencil sketch effect\n");
		printf(" 9 - Psychedelic glitch effect\n");
		printf(" 10 - Negative with red edges\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testSepiaEffect();
				break;
			case 2:
				testRainbowEffect();
				break;
			case 3:
				testMosaicEffect();
				break;
			case 4:
				testOilPaintEffect();
				break;
			case 5:
				testRetricaEffect();
				break;
			case 6:
				test3DAnaglyphEffect();
				break;
			case 7:
				testCrossWavesPattern();
				break;
			case 8:
				testPencilSketchEffect();
				break;
			case 9:
				testPsychedelicGlitchEffect();
				break;
			case 10:
				testNegativeWithRedEdgesEffect();
				break;
		}
	}
	while (op!=0);
	return 0;
}