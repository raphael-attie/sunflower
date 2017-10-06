void cbilin_interp(float *image, float *xout, float *yout, float *values, int nx, int npts)
{
    unsigned int x0, y0;
	for (int k=0; k<npts; k++){

            x0 = (unsigned int)xout[k];
            y0 = (unsigned int)yout[k];
            // Get the 4 nearest neighbours
            float h00, h01, h10, h11;
            // Left
            h00 = image[y0*nx + x0];
            // Right
            h10 = image[y0*nx + x0 + 1];
            // Bottom?
            h01 = image[(y0+1)*nx + x0];
            // Top?
            h11 = image[(y0+1)*nx + x0 + 1];

            // Calculate the weights for each pixel
            float fx = xout[k] - (float)x0;
            float fy = yout[k] - (float)y0;
            float fx1 = 1.0f - fx;
            float fy1 = 1.0f - fy;

            float w1 = fx1 * fy1;
            float w2 = fx * fy1;
            float w3 = fx1 * fy;
            float w4 = fx * fy;

            // Calculate the weighted sum of pixels
            values[k] = h00 * w1 + h10 * w2 + h01 * w3 + h11 * w4;
        }

        return;
}

void cbilin_interp2d(float *image, float *xout, float *yout, float *values, int nx, int n1, int n2)
{
    unsigned int x0, y0;
	for (int j=0; j<n1; j++){
	    for (int i=0; i<n2; i++){
	        x0 = (unsigned int)xout[j*n2 + i];
            y0 = (unsigned int)yout[j*n2 + i];
            // Get the 4 nearest neighbours
            float h00, h01, h10, h11;
            // Left
            h00 = image[y0*nx + x0];
            // Right
            h10 = image[y0*nx + x0 + 1];
            // Bottom?
            h01 = image[(y0+1)*nx + x0];
            // Top?
            h11 = image[(y0+1)*nx + x0 + 1];

            // Calculate the weights for each pixel
            float fx = xout[j*n2 + i] - (float)x0;
            float fy = yout[j*n2 + i] - (float)y0;
            float fx1 = 1.0f - fx;
            float fy1 = 1.0f - fy;

            float w1 = fx1 * fy1;
            float w2 = fx * fy1;
            float w3 = fx1 * fy;
            float w4 = fx * fy;

            // Calculate the weighted sum of pixels
            values[j*n2 + i] = h00 * w1 + h10 * w2 + h01 * w3 + h11 * w4;
	    }
    }

        return;
}