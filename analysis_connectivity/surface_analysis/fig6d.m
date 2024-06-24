addpath /Users/ulises/Documents/MATLAB/gifti
recording_times = [50];
num_recording_times = length(recording_times);
rates_file = zeros(num_recording_times,89, 2);
num_areas = size(rates_file,2);
% load in left hemisphere inflated surface
l_inflated = gifti('surface_files/MacaqueYerkes19.L.inflated.32k_fs_LR.surf.gii');
% load in LH kennedy atlas (91 regions)
kennedy_atlas_91 = gifti('surface_files/kennedy_atlas_91.label.gii');


%%
% load in a gifti file of the right type in order to get a
% template to write over
example = gifti('surface_files/cortical_thickness.func.gii');
num_vertices = length(example.cdata);
example.cdata = zeros(num_vertices,1);
%%
% get area List in Donahue order
% areaList_Donahue = kennedy_atlas_91.labels.name(2:end)';
% Note 18-Jan-2021 - for some reason the labels.name method has stopped
% correctly reading the label file.
load surface_files/areaList_Donahue.mat
load surface_files/names_ctx.mat
load surface_files/Fig6/overlaps_salient_random_visual.mat
load surface_files/Fig6/mean_rate_salient_random_visual.mat


[~, ulises_areas_in_Donahue_idx] = ismember(names_ctx,areaList_Donahue);

rate_map = example;
overlap_map = example;





for current_time = 1:num_recording_times
    
for current_parcel = 1:num_areas
    vertices_in_parcel = find(kennedy_atlas_91.cdata==ulises_areas_in_Donahue_idx(current_parcel)); % note kennedy_atlas_91.cdata ranges from 0-91, not 1-92
    overlap_map.cdata(vertices_in_parcel,current_time) = overlaps_salient_random_visual(recording_times(current_time),current_parcel, 3);
    rate_map.cdata(vertices_in_parcel,current_time) = rate_salient_random_visual(current_parcel, recording_times(current_time));
end

end


fig = plot(l_inflated, overlap_map);
colormap(flipud(hot(10)));
c = colorbar('east');
c.Position = [0.75, 0.18, 0.05, 0.25];
clim([0, 0.45]) %mean rate
%clim([0, 0.4]) %overlap
material([0.3, 0.6, 0])
%colormapeditor