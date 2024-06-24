addpath /Users/ulises/Documents/MATLAB/gifti
recording_times = [533];
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
load surface_files/fig4/mean_rate_fpn_asym.mat
load surface_files/fig4/overlaps_fpn_asym.mat


[~, ulises_areas_in_Donahue_idx] = ismember(names_ctx,areaList_Donahue);

rate_map = example;
ov_fpn_8 = example;
ov_fpn_6 = example;

ind_meas = 200;

    
for current_parcel = 1:num_areas
    vertices_in_parcel = find(kennedy_atlas_91.cdata==ulises_areas_in_Donahue_idx(current_parcel)); % note kennedy_atlas_91.cdata ranges from 0-91, not 1-92
    ov_fpn_8.cdata(vertices_in_parcel,1) = overlaps_fpn_asym(67,current_parcel, 8);
    ov_fpn_6.cdata(vertices_in_parcel,1) = overlaps_fpn_asym(67,current_parcel, 5);
    rate_map.cdata(vertices_in_parcel,1) = rate_fpn_asym(current_parcel,67);

end




fig = plot(l_inflated, rate_map);
colormap(flipud(hot(10)));
c = colorbar('south');
c.Position = [0.275, 0.18, 0.5, 0.05];
clim([0, 0.25])
material([0.3, 0.6, 0])
%colormapeditor