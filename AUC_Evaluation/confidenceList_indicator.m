function [conf_list_good, conf_list_other] = confidenceList_indicator(im_conf,im_GT,im_indi,sort_direction)

if ~isempty(im_indi)
    % get pixels of good region
    [row_good, col_good] = find(im_indi==1);
    conf_list_good = im_conf(im_indi==1);
    conf_list_good = [row_good, col_good, conf_list_good];

    % get pixels of not good region
    [row_other, col_other] = find(im_GT~=0 & im_indi==0);
    conf_list_other = im_conf(im_GT~=0 & im_indi==0);
    conf_list_other = [row_other, col_other, conf_list_other];

    % sort confidence list
    conf_list_good = sortrows(conf_list_good,3, sort_direction);
    conf_list_other = sortrows(conf_list_other,3, sort_direction);
else
    conf_list_other = false;
    % get the coordinates of valid pixels from ground truth map, which indicates the values of them are not 0
    [row,col] = find(im_GT~=0);
    % quary these pixels from the corresponding confidence map
    conf_list = im_conf(im_GT~=0);
    % get a n*3 confidence list (matrix), including coordinates (x,y) of the pixel and the confidence value  
    conf_list = [row col conf_list]; 
    % sort the confidence list according to 3rd column (confidence level), descent
    conf_list_good = sortrows(conf_list, 3, sort_direction); 

end