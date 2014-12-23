function selective_search_kitti_precomp(imdb)
% computes  & caches selective search boxes for 
% imdb: the given image database
regions_file = sprintf('./data/selective_search_data/%s', imdb.name);

regions.boxes = {};
for i = 1:length(imdb.image_ids)
    image_path = imdb.image_at(i)
    image = imread(image_path);
    boxes = selective_search_boxes(image, true, 1000);
    regions.boxes{i} = boxes;
%     figure(1),
%     imshow(image);
%     hold on
%     axis equal;
%     for j = 1:5: length(boxes)
%         box = boxes(j,[2 1 4 3]);
%         h = rectangle('Position', [box(1),box(2),box(3)-box(1),box(4)-box(2)], 'EdgeColor', rand(1,3));
%     end
%     pause;
%     hold off
end

save(regions_file, regions); 