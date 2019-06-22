# SAMPLES=5000
SAMPLES=1
# THREADS=4
THREADS=1
for theta_i in 1e-5 1 3
do
    for f_a in 1e12 1e17
    do
        python3 error_propagation.py $SAMPLES $THREADS $theta_i $f_a "../Data/theta_i_${theta_i}_f_a_${f_a}.txt"
    done
done

