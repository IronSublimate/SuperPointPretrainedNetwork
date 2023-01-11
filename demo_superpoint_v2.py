import numpy as np
import os
import time

import cv2

from demo_superpoint import PointTracker
from utils.args_parser import args_parser
from utils.frontend import SuperPointFrontend
from utils.video_streamer import VideoStreamer

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3:  # pragma: no cover
    print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0., 0., 0.5],
                  [0., 0., 0.99910873],
                  [0., 0.37843137, 1.],
                  [0., 0.83333333, 1.],
                  [0.30044276, 1., 0.66729918],
                  [0.66729918, 1., 0.30044276],
                  [1., 0.90123457, 0.],
                  [1., 0.48002905, 0.],
                  [0.99910873, 0.07334786, 0.],
                  [0.5, 0., 0.]])

if __name__ == '__main__':

    opt = args_parser()
    # This class helps load input images from different sources.
    vs = VideoStreamer(opt.input, opt.camid, opt.H, opt.W, opt.skip, opt.img_glob)

    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(weights_path=opt.weights_path,
                            nms_dist=opt.nms_dist,
                            conf_thresh=opt.conf_thresh,
                            nn_thresh=opt.nn_thresh,
                            cuda=opt.cuda,
                            use_trt=opt.trt)
    print('==> Successfully loaded pre-trained network.')

    # This class helps merge consecutive point matches into tracks.
    tracker = PointTracker(opt.max_length, nn_thresh=fe.nn_thresh)

    # Create a window to display the demo.
    if not opt.no_display:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4

    # Create output directory if desired.
    if opt.write:
        print('==> Will write outputs to %s' % opt.write_dir)
        if not os.path.exists(opt.write_dir):
            os.makedirs(opt.write_dir)

    print('==> Running Demo.')
    total_time = 0.0
    cnt = 0
    while True:
        cnt += 1
        start = time.time()

        # Get a new image.
        img, status = vs.next_frame()
        if status is False:
            break

        # Get points and descriptors.
        start1 = time.time()
        pts, desc, heatmap = fe.run(img)
        end1 = time.time()
        total_time += (end1 - start1)
        # Add points and descriptors to the tracker.
        tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = tracker.get_tracks(opt.min_length)

        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(fe.nn_thresh)  # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        if opt.show_extra:
            cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

        # Extra output -- Show current point detections.
        out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
            out3 = myjet[np.round(np.clip(heatmap * 10, 0, 9)).astype('int'), :]
            out3 = (out3 * 255).astype('uint8')
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

        # Resize final output.
        if opt.show_extra:
            out = np.hstack((out1, out2, out3))
            out = cv2.resize(out, (3 * opt.display_scale * opt.W, opt.display_scale * opt.H))
        else:
            out = cv2.resize(out1, (opt.display_scale * opt.W, opt.display_scale * opt.H))

        # Display visualization image to screen.
        if not opt.no_display:
            cv2.imshow(win, out)
            key = cv2.waitKey(opt.waitkey) & 0xFF
            if key == ord('q'):
                print('Quitting, \'q\' pressed.')
                break

        # Optionally write images to disk.
        if opt.write:
            out_file = os.path.join(opt.write_dir, 'frame_%05d.png' % vs.i)
            print('Writing image to %s' % out_file)
            cv2.imwrite(out_file, out)

        end = time.time()
        net_t = (1. / float(end1 - start))
        total_t = (1. / float(end - start))
        if opt.show_extra:
            print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).' \
                  % (vs.i, net_t, total_t))

    # Close any remaining windows.
    cv2.destroyAllWindows()

    print("average time", total_time / cnt)
    print('==> Finshed Demo.')
