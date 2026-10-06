function set_pd(new_pd)
    load('bgn_vars.mat');
    pd = double(new_pd);
    save('bgn_vars.mat');
end
