void cbilin_interp(double *image, double *xout, double *yout, double *values, int nx, int npts)
{
    unsigned int x0, y0;
	for (int k=0; k<npts; k++){

            x0 = (unsigned int)xout[k];
            y0 = (unsigned int)yout[k];
            // Get the 4 nearest neighbours
            double h00, h01, h10, h11;
            // Left
            h00 = image[y0*nx + x0];
            // Right
            h10 = image[y0*nx + x0 + 1];
            // Bottom?
            h01 = image[(y0+1)*nx + x0];
            // Top?
            h11 = image[(y0+1)*nx + x0 + 1];

            // Calculate the weights for each pixel
            double fx = xout[k] - (double)x0;
            double fy = yout[k] - (double)y0;
            double fx1 = 1.0f - fx;
            double fy1 = 1.0f - fy;

            double w1 = fx1 * fy1;
            double w2 = fx * fy1;
            double w3 = fx1 * fy;
            double w4 = fx * fy;

            // Calculate the weighted sum of pixels
            values[k] = h00 * w1 + h10 * w2 + h01 * w3 + h11 * w4;
        }

        return;
}