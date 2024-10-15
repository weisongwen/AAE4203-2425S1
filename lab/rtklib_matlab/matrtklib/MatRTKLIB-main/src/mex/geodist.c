/**
 * @file geodist.c
 * @brief Geometric distance and receiver-to-satellite unit vector
 * @author Taro Suzuki
 * @note Wrapper for "geodist" in rtkcmn.c
 * @note Support vector inputs
 */

#include "mex_utility.h"

#define NIN 4

/* mex interface */
extern void mexFunction(int nargout, mxArray *argout[], int nargin,
                        const mxArray *argin[]) {
    int i, j, m, nsat, nrrs;
    double rr[3], rs[3], e[3], *rrs, *rsx, *rsy, *rsz, *d, *ex, *ey, *ez;

    /* check arguments */
    mxCheckNumberOfArguments(nargin, NIN);
    mxCheckSameSize(argin[0], argin[1]); /* rsx,rsy */
    mxCheckSameSize(argin[1], argin[2]); /* rsy,rsz */
    mxCheckSizeOfColumns(argin[3], 3);   /* rrs */

    /* inputs */
    rsx = (double *)mxGetPr(argin[0]);
    m = (int)mxGetM(argin[0]);
    rsy = (double *)mxGetPr(argin[1]);
    nsat = (int)mxGetN(argin[0]);
    rsz = (double *)mxGetPr(argin[2]);
    rrs = (double *)mxGetPr(argin[3]);
    nrrs = (int)mxGetM(argin[3]);

    // mexPrintf("m=%d nsat=%d\n",m,nsat);

    /* outputs */
    argout[0] = mxCreateDoubleMatrix(m, nsat, mxREAL);
    d = mxGetPr(argout[0]);
    argout[1] = mxCreateDoubleMatrix(m, nsat, mxREAL);
    ex = mxGetPr(argout[1]);
    argout[2] = mxCreateDoubleMatrix(m, nsat, mxREAL);
    ey = mxGetPr(argout[2]);
    argout[3] = mxCreateDoubleMatrix(m, nsat, mxREAL);
    ez = mxGetPr(argout[3]);

    /* call RTKLIB function */
    for (i = 0; i < m; i++) {
        if (nrrs == 1) {
            rr[0] = rrs[0];
            rr[1] = rrs[1];
            rr[2] = rrs[2];
        } else {
            rr[0] = rrs[i + m * 0];
            rr[1] = rrs[i + m * 1];
            rr[2] = rrs[i + m * 2];
        }
        // mexPrintf("i=%d rrx:%.1f rry:%.1f rrz:%.1f \n",i,rr[0],rr[1],rr[2]);
        for (j = 0; j < nsat; j++) {
            rs[0] = rsx[i + m * j];
            rs[1] = rsy[i + m * j];
            rs[2] = rsz[i + m * j];
            if (mxIsNaN(rr[0]) || mxIsNaN(rr[1]) || mxIsNaN(rr[2])) {
                d[i + m * j] = ex[i + m * j] = ey[i + m * j] = ez[i + m * j] = mxGetNaN();
                continue;
            }
            d[i + m * j] = geodist(rs, rr, e);
            ex[i + m * j] = e[0];
            ey[i + m * j] = e[1];
            ez[i + m * j] = e[2];
        }
    }
}
