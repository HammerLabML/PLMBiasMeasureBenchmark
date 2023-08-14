config="configs/test.yaml"

if python3 initialize_experiment.py -c $config; then
    echo "initialization and tests successful -> start slurm jobs"
    #run sbatch
else
    echo "initialization or tests failed"
fi

