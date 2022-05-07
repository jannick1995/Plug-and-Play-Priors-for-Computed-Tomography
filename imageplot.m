function imageplot(im,clims)

if nargin < 2
    imagesc(im)
else
    imagesc(im,clims)
end

colormap gray
axis image, axis off

end