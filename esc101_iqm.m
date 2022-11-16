% Measure the quality of blurred images through foggy glass
% Paper: https://www.sciencedirect.com/science/article/pii/S1877705813016007/pdf?md5=373f56b11195cbee697c05de5debe6b6&pid=1-s2.0-S1877705813016007-main.pdf
% You should run this code as a MATLAB live script

disp("very foggy");
image_quality("https://glassrepairsingapore.com/wp-content/uploads/dreamstime_m_83632770_min-1024x683.jpg");

disp("median foggy");
image_quality("https://lindleysautocentres.co.uk/wp-content/uploads/2019/12/car-windscreen-condensation-scaled-e1576583761453.jpg");

disp("clear");
image_quality("https://media.istockphoto.com/id/494851970/photo/rocky-mountain-road-trip.jpg?s=612x612&w=0&k=20&c=yK_9lUHS_5MRYEIwaL6yKPJGPrd7Uhq3sKjgMD5f1B8=");



function [] = image_quality(path)

    % load image, resize to 256x256 for consistency
    img = imread(path);
    img = imresize(img, [256, 256]);
    % https://en.wikipedia.org/wiki/YUV#Conversion_to/from_RGB
    i = 0.299*img(:,:,1)+0.587*img(:,:,2)+0.114*img(:,:,3);

    % follow the paper instruction
    af = fftshift(abs(fft2(i))) / sqrt(numel(i));
    m = max(af, [], 'all');
    th = af > (m/1000);
    iqm = sum(th, 'all') / numel(i);

    % visualization
    disp("Quality: " + iqm);
    figure
    imshow(i, []);
    figure
    image(af);
    colorbar;
end

