#include <iostream>
#include <unordered_set>
#include <cstdio>
#include "opencv2/opencv.hpp"

#define PI 3.14159

using namespace cv;
using namespace std;

struct Sline {
	//represents a line via 2 Points
	Point a, b;
	Sline() {
		a = Point(0, 0);
		b = Point(0, 0);
	}
	Sline(const Point& p1, const Point& p2) : a(p1) , b(p2) {}
};

struct Circle {
	//represents a circle via centre and radius
	Point centre;
	int radius;

	Circle() {
		centre = Point(0, 0);
		radius = 0;
	}

	Circle(const Point& p, const int& r) : centre(p), radius(r) {}

	Circle(Point p1, Point p2, Point p3) {
		double A = p1.x * (p2.y - p3.y) - p1.y*(p2.x - p3.x) + p2.x * p3.y - p3.x * p2.y;
		double B = (p1.x * p1.x + p1.y * p1.y) * (p3.y - p2.y) + (p2.x * p2.x + p2.y * p2.y) * (p1.y - p3.y)
			+ (p3.x * p3.x + p3.y * p3.y) * (p2.y - p1.y);
		double C = (p1.x * p1.x + p1.y * p1.y) * (p2.x - p3.x) + (p2.x*p2.x + p2.y*p2.y) * (p3.x - p1.x)
			+ (p3.x*p3.x + p3.y*p3.y) * (p1.x - p2.x);

		int x = -B / (2 * A);
		int y = -C / (2 * A);
		centre = Point(x, y);
		radius = sqrt(pow(x - p1.x, 2) + pow(y - p1.y, 2));
	}
};

int dist2Circle(const Circle& c, const Point& p) {
	return abs(sqrt(pow(c.centre.x - p.x, 2) + pow(c.centre.y - p.y, 2)) - c.radius);
}

int normalDist(const Sline& l, const Point& p) {
	int x0 = p.x, y0 = p.y, x1 = l.a.x, y1 = l.a.y, x2 = l.b.x, y2 = l.b.y;
	int numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2 * y1 - y2 * x1);
	int denominator = sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2));
	return (int)numerator / denominator;
}

int getMean(const Mat &image) {
	int sum = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			sum += (int)image.at<uchar>(i, j);
		}
	}
	return (int)(sum / (image.rows * image.cols));
}

vector<Point> pickNRandom(const vector<Point>& points, int n) {
	vector<Point> results;
	//const unsigned long dist = points.size();
	//const unsigned long divisor = (RAND_MAX + 1) / dist;
	//unsigned long k;

	while (results.size() < n) {
		//do { k = rand() / divisor; } while (k >= dist);
		int index = rand() % points.size();
		Point p = points[index];
		if(find(results.begin(), results.end(), p) == results.end())
			results.push_back(p);
	}

	return results;
}

vector<Circle> circleRansac(const Mat& edges, int thresh, int max_iter, float ratio = 0.7) {
	//store points
	vector<Point> points;
	vector<Circle> results;
	int total_points = 0;

	for(int row = 0; row < edges.rows; row++) {
		for (int col = 0; col < edges.cols; col++) {
			if (edges.at<uchar>(row, col)) {
				points.push_back(Point(col, row));
				total_points++;
			}
		}
	} //end for

	int points_seen = 0;
	int limit = ratio * total_points;
	while (points_seen < limit) {
		Circle best_circle;
		vector<Point> inliers, best_inliers;
		int best_count = 0;

		for (int i = 0; i < max_iter; i++) {
			//pick 3 random points
			vector<Point> three_random = pickNRandom(points, 3);
			Circle circ = Circle(three_random[0], three_random[1], three_random[2]);

			//find all inlier points
			for (int j = 0; j < points.size(); j++) {
				Point p = points[j];
				if (dist2Circle(circ, p) <= thresh) {
					inliers.push_back(p);
				}
			} // end for

			if (inliers.size() > best_count) {
				best_count = inliers.size();
				best_circle = circ;
				best_inliers = inliers;
			}

			inliers.clear();
		} //end for max_iter

		results.push_back(best_circle);
		//update points_seen
		points_seen += best_count;
		//remove inliers from points set
		vector<Point>::const_iterator it;
		for (int k = 0; k < best_inliers.size(); k++) {
			it = points.begin();
			Point p = best_inliers[k];
			for (int i = 0; i < points.size(); i++) {
				if (points[i] == p) {
					advance(it, i);
					points.erase(it);
					break;
				}
			}
		}
	}//end while points_seen
	return results;
}

vector<Sline> lineRansac(const Mat& edges, int thresh, int max_iter, float ratio=0.7) {
	//store points in a set
	vector<Point> points;
	vector<Sline> results;
	int total_points = 0;

	for (int row = 0; row < edges.rows; row++) {
		for (int col = 0; col < edges.cols; col++) {
			if (edges.at<uchar>(row, col)) {
				points.push_back(Point(col, row));
				total_points++;
			}
		}
	} //end for

	int points_seen = 0;
	while (points_seen < ratio * total_points) {
		Sline best_line;
		vector<Point> inliers, best_inliers;
		int best_count = 0;

		for (int i = 0; i < max_iter; i++) {
			//pick 2 random points
			vector<Point> two_random = pickNRandom(points, 2);
			Sline line = Sline(two_random[0], two_random[1]);
			
			//find all the inlier points
			for (int j = 0; j < points.size(); j++) {
				Point p = points[j];
				if (normalDist(line, p) <= thresh) {
					inliers.push_back(p);
				}
			}

			//decide if this line is the best
			if (inliers.size() > best_count) {
				best_line = line;
				best_count = inliers.size();
				best_inliers = inliers;
			}

			inliers.clear();

		} // end max_iter loop
		//best line after max_iter iterations found
		//add line to results
		results.push_back(best_line);
		//update points_seen
		points_seen += best_count;
		//remove inliers from points set
		vector<Point>::const_iterator it;
		for (int k = 0; k < best_inliers.size();k++) {
			it = points.begin();
			Point p = best_inliers[k];
			for (int i = 0; i < points.size(); i++) {
				if (points[i] == p) {
					advance(it, i);
					points.erase(it);
					break;
				}
			}
		}

	} //end while
	return results;
}

void drawLines(Mat& image, const vector<Sline>& lines) {
	float m;
	int b;
	int x0, x1, y0, y1;
	//int y_coordinate;
	Scalar color = Scalar(0, 0, 255);

	for (int i = 0; i < lines.size(); i++) {
		Sline l = lines[i];
		m = (l.b.y - l.a.y) / max(0.000000001, (double)(l.b.x - l.a.x));
		b = l.a.y - m * l.a.x;

		y0 = b;
		x0 = 0;
		if (y0 > image.rows - 1) {
			y0 = image.rows - 1;
		}
		else if (y0 < 0) {
			y0 = 0;
		}
		x0 = (y0 - b) / m;

		x1 = image.cols - 1;
		y1 = m * x1 + b;
		if (y1 > image.rows - 1) {
			y1 = image.rows - 1;
		}
		else if (y1 < 0) {
			y1 = 0;
		}
		x1 = (y1 - b) / m;

		line(image, Point(x0, y0), Point(x1, y1), color);
	}//end for
}

void drawLines2(Mat& image, const vector<Sline>& lines) {
	float m;
	int b;
	int y;
	//int x, y;
	//int y_coordinate;
	//Scalar color = Scalar(0, 0, 255);

	for (int i = 0; i < lines.size(); i++) {
		Sline l = lines[i];
		m = (l.b.y - l.a.y) / max(0.000000001, (double)(l.b.x - l.a.x));
		b = l.a.y - m * l.a.x;

		for (int x = 0; x < image.cols; x++) {
			y = m * x + b;
			if (y >= 0 && y < image.rows) {
				image.at<Vec3b>(Point(x,y))[0] = 0;
				image.at<Vec3b>(Point(x, y))[1] = 0;
				image.at<Vec3b>(Point(x, y))[2] = 255;
			}
		}
	}//end for
}

void drawCircle(Mat& image, const vector<Circle>& circles) {
	for (Circle c : circles) {
		for (double theta = 0.0; theta < 2 * PI; theta += 0.001) {
			int x = c.centre.x + c.radius * cos(theta);
			int y = c.centre.y + c.radius * sin(theta);
			if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
				image.at<Vec3b>(Point(x, y))[0] = 0;
				image.at<Vec3b>(Point(x, y))[1] = 0;
				image.at<Vec3b>(Point(x, y))[2] = 255;
			}
		}
	}
}

int main(int argc, char **argv)
{
	srand(69);

	Mat colorImage, greyImage, edges;
	colorImage = imread("images\\concentric_circles.jpg");
	cvtColor(colorImage, greyImage, COLOR_BGR2GRAY);
	//blur(greyImage, greyImage, Size(2, 2));
	int avg = getMean(greyImage);
	int lowerThresh = (int)avg * 0.5;
	int upperThresh = (int)avg * 1.25;

	Canny(greyImage, edges, lowerThresh, upperThresh);

	//vector<Sline> lines = lineRansac(edges, 3, 100);
	//drawLines(colorImage, lines);

	vector<Circle> circles = circleRansac(edges, 3, 100);
	drawCircle(colorImage, circles);

	namedWindow("seaside");
	imshow("seaside", edges);

	namedWindow("seaside-lines");
	imshow("seaside-lines", colorImage);

	waitKey(0);

	return 0;
}
