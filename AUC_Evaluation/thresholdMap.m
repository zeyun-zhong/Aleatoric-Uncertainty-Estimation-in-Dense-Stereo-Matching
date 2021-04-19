function [T] = thresholdMap(threshold,im_GT)

[m,n] = size(im_GT);
if length(threshold) == 2
    T = min(threshold) * im_GT; % creating the threshold matrix, min(threshold) is the percentage threshold 
    idx = T <= max(threshold); % max(threshold) is the absolute pixel value threshold
    T(idx) = max(threshold); % assign the absoulte pixel value threshold to threshold map
else
    if threshold >= 1
        T = threshold * ones(m,n);
    else
        T = threshold * im_GT;
    end
end

end

