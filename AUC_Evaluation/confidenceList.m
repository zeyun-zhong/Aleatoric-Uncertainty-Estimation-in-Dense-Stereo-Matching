function [conf_list] = confidenceList(im_conf,im_GT, sort_direction)

% get the coordinates of valid pixels from ground truth map, whcih indicates the values of them are not 0
[row,col] = find(im_GT~=0);
% quary these pixels from the corresponding confidence map
conf_list = im_conf(im_GT~=0);
% get a n*3 confidence list (matrix), including coordinates (x,y) of the pixel and the confidence value  
conf_list = [row col conf_list]; 
% sort the confidence list according to 3rd column (confidence level), descent
conf_list = sortrows(conf_list,3, sort_direction); 

end

