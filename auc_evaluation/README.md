1. Introduction

    auc calculate[1] in parallel base on MPI.

2. How to:

    step 1:

        split your predict data into n parts. (n equal the number of nodes of your cluster)

    step 2:

        change Ip in the file base on your ips and then rum commend: sh run.sh


3. Contact:

    2012wxs@gmail.com

4. References:

    [1]An introduction to ROC analysis, http://dl.acm.org/citation.cfm?id=1159475&preflayout=flat


About AUC calculate:

    1, ROC：a cluster of dots in two-dimension coordinates, x-axis is fp rate = FP/N, y-axis is tp rate = TP/P. 
    
    2, AUC: area under the ROC curve. in our code, I inore the denominator, as it the x-asis is scaled by N and y-axis is scaled by P, and the AUC result is also correct.  
