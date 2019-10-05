/*
 * matrix.h
 */

//NOTE NOT SELF-MADE CODE
//MADE BY
//https://github.com/akalicki/matrix

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>
#include <vector>
class Matrix {
    public:
        Matrix(int, int);
        Matrix();
        ~Matrix();
        Matrix(const Matrix&);
        Matrix& operator=(const Matrix&);

        inline double& operator()(int x, int y) { return p[x][y]; }

        Matrix& operator+=(const Matrix&);
        Matrix& operator-=(const Matrix&);
        Matrix& operator*=(const Matrix&);
        Matrix& operator*=(double);
        Matrix& operator/=(double);
        Matrix  operator^(int);
        
        friend std::ostream& operator<<(std::ostream&, const Matrix&);
        friend std::istream& operator>>(std::istream&, Matrix&);

        void swapRows(int, int);
        Matrix transpose();

        static Matrix createIdentity(int);
        static Matrix solve(Matrix, Matrix);
        static Matrix bandSolve(Matrix, Matrix, int);
		static Matrix fromVector(std::vector<double> arr) {
			Matrix m = Matrix(arr.size(), 1);
			for (size_t i = 0; i < arr.size(); i++) {
				m.p[i][0] = arr[i];
			}
			return m;
		}
		static std::vector<double> toVector(Matrix arr) {
			std::vector<double> m;
			for (int i = 0; i < arr.rows_; i++) {
				for (int j = 0; j < arr.cols_; j++) {
					m.push_back(arr.p[i][j]);
				}
			}
			return m;
		}
        // functions on vectors
        static double dotProduct(Matrix, Matrix);

        // functions on augmented matrices
        static Matrix augment(Matrix, Matrix);
        Matrix gaussianEliminate();
        Matrix rowReduceFromGaussian();
        void readSolutionsFromRREF(std::ostream& os);
        Matrix inverse();
		void map(double(*f)(double));
		void Randomize();
		
    private:
        int rows_, cols_;
		double **p;

        void allocSpace();
        Matrix expHelper(const Matrix&, int);
};

Matrix operator+(const Matrix&, const Matrix&);
Matrix operator-(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, double);
Matrix operator*(double, const Matrix&);
Matrix operator/(const Matrix&, double);

#endif
