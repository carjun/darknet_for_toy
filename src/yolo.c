#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

//char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
//image voc_labels[20];
char *voc_names[] = {"helmet", "house", "blue car", "rose", "elephant", "snowman", "rabbit", "spongebob", "turtle", "gavel", "ladybug", "praying mantis", "green car", "saw", "doll", "phone", "rubik\'s cube", "rake", "truck", "white car", "lady bug rattle", "cube", "bed", "cube with ball"};
image voc_labels[24];

void train_yolo(char *cfgfile, char *weightfile)
{
    //char *train_images = "/data/voc/train.txt";
     //char *train_images = "/l/vision/v3/zehzhang/train.txt";
     char *train_images = "/l/vision/v3/zehzhang/toy_data/train_toy_2.txt";
    //char *backup_directory = "/home/pjreddie/backup/";
     //char *backup_directory = "/l/vision/v3/zehzhang/backup_weights/";
     char *backup_directory = "/l/vision/v3/zehzhang/toy_data/backup_weights_toy_2/";
      //char *backup_directory = "/l/vision/v3/zehzhang/toy_data/backup_weights_toy_restart/";
     //char *backup_directory = "/l/vision/v3/zehzhang/toy_data/backup_weights_toy/";
     //char *backup_directory = "/l/vision/v3/zehzhang/toy_data/backup_weights_test/";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);

void convert_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];//scale mean confidence score for each box.
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];//for each grid cell, the probabilities of each class are computed and stored in the array of "predictions". 
                                                                       //so P(an object belonging to a certain class is in a certain box) = scale * predictions
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}



//for more detailed explanation for this part, take a look at the file å¯¹yoloçš„ä¸€äº›å‚æ•°è§£ï¿½?pdf
//convert_detections_toy is for toy data.
//Considering there is at most 1 toy of each class in a image, among many boxes predicting one same class, we only choose the one with the greatest probabilies
void convert_detections_toy(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n,k;
     float *probs_max = calloc(classes, sizeof(float *));
     int *max_index = calloc(classes, sizeof(int *));
     for(k = 0; k < classes; k++) probs_max[k] = 0;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];//scale mean confidence score for each box.
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];//for each grid cell, the probabilities of each class are computed and stored in the array of "predictions". 
                                                                       //so P(an object belonging to a certain class is in a certain box) = scale * predictions
                if (prob > probs_max[j]){
                      probs[index][j] = (prob > thresh) ? prob : 0;// this thresh is prob_thresh
                      probs_max[j] = prob;
                      probs[max_index[j]][j] = 0;
                      max_index[j] = index;}
                  else probs[index][j] = 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}




void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)//to generate some files in PASCAL VOC standard.
                                                                                                                 // These files can be processed by the scoring
																												 //code of PASCAL VOC Challenge to generate 
																												 //precision and recall graph 
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],//Here it writes the information of every box with a probability of this 
			                                                                        //class greater than 0 to the file. Of course we can set a threshold to
			                                                                        //do this.
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/test_toy.txt");
    list *plist = get_paths("data/video_parent_list.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

//    float thresh = .001;
    float thresh = .001;//this threshold value is for picking region proposals with a confidence score (which is exactly scale in this script) greater than the threshold
    float iou_thresh = .5;
//    float iou_thresh = .8;
//    float nms = 0;
    float nms = 0.4;
    float thresh_for_prob = 0.1521;//this threshold value is for picking region proposals with a probabilities of a certain class greater than the threshold
                                //since we use convert_detections_toy, we can make this value equal to 0
    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 1);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
        convert_detections_toy(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 0);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        draw_detections(orig, l.side*l.side*l.n, thresh_for_prob, boxes, probs, voc_names, voc_labels, 24);//I add this to draw detections in each test image.
        // draw_detections function is in image.c
//		char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_parent/JPEGImages/", "testF/parent/");
//         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_child/JPEGImages/", "testF/child/");
		char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/video_parent/20150530_16859_sync_frames_parent/JPEGImages/", "/l/vision/v3/zehzhang/toy_data/video_result/parent/");
         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/video_child/20150530_16859_sync_frames_child/JPEGImages/", "/l/vision/v3/zehzhang/toy_data/video_result/child/");
        save_image(orig, imagepath);


        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);// the thresh in fact should be thresh_for_prob, 
		                                                                                           // but since the last argument is 1 and thus the code will 
																								   // return confidence score, so what number you put there 
																								   // does not matter

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);//For details about box_label and read_boxes, take a look at data.c and data.h
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
							// threshold as the proposals
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
            }
        }

        fprintf(stderr, "%5d %5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, proposals, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
		// RPs/Img: region proposals per image. The recall above means the recall of each bounding boxes because when getting the correct predictions, it only
		// consider the boxes whose confidence score and IOU are greater than their thresholds respectively as the correct predictions. 
		// In other words, it does not consider whether the program predict the right class of the object. 
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void validate_yolo_recall_video(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/test_toy.txt");
    list *plist = get_paths("/l/vision/v3/sbambach/_postdoc/marr/exp12/__20161022_17878/child_data.txt");
//    list *plist = get_paths("/l/vision/v3/sbambach/_postdoc/marr/exp12/__20161022_17878/parent_data.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;//, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

//    float thresh = .001;
    float thresh = .001;//this threshold value is for picking region proposals with a confidence score (which is exactly scale in this script) greater than the threshold
    float iou_thresh = .5;
//    float iou_thresh = .8;
//    float nms = 0;
    float nms = 0.4;
    float thresh_for_prob = 0.1521;
//    float thresh_for_prob = 0.1117;//this threshold value is for picking region proposals with a probabilities of a certain class greater than the threshold
                                //since we use convert_detections_toy, we can make this value equal to 0
    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 1);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
        convert_detections_toy(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 0);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        draw_detections(orig, l.side*l.side*l.n, thresh_for_prob, boxes, probs, voc_names, voc_labels, 24);//I add this to draw detections in each test image.
        // draw_detections function is in image.c
//		char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_parent/JPEGImages/", "testF/parent/");
//         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_child/JPEGImages/", "testF/child/");
        char *imagepath = find_replace(path, "cam07_frames_p", "child_results_1");
        imagepath = find_replace(imagepath, "cam08_frames_p", "parent_results_2");
         
        save_image(orig, imagepath);

/*
        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);// the thresh in fact should be thresh_for_prob, 
		                                                                                           // but since the last argument is 1 and thus the code will 
																								   // return confidence score, so what number you put there 
																								   // does not matter

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);//For details about box_label and read_boxes, take a look at data.c and data.h
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
							// threshold as the proposals
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
            }
        }

        fprintf(stderr, "%5d %5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, proposals, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
		// RPs/Img: region proposals per image. The recall above means the recall of each bounding boxes because when getting the correct predictions, it only
		// consider the boxes whose confidence score and IOU are greater than their thresholds respectively as the correct predictions. 
		// In other words, it does not consider whether the program predict the right class of the object. 
*/
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void validate_yolo_recall_coordinates(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/test_toy.txt");
    list *plist = get_paths("/l/vision/v3/sbambach/_postdoc/marr/exp12/__20161022_17878/child_data.txt");
//    list *plist = get_paths("/l/vision/v3/sbambach/_postdoc/marr/exp12/__20161022_17878/parent_data.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

//    float thresh = .001;
    float thresh = .001;//this threshold value is for picking region proposals with a confidence score (which is exactly scale in this script) greater than the threshold
    float iou_thresh = .5;
//    float iou_thresh = .8;
//    float nms = 0;
    float nms = 0.4;
    float thresh_for_prob = 0.1521;
//    float thresh_for_prob = 0.1117;//this threshold value is for picking region proposals with a probabilities of a certain class greater than the threshold
                                //since we use convert_detections_toy, we can make this value equal to 0
    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 1);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
        convert_detections_toy(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 0);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);
        
		char *imagepath = find_replace(path, "/l/vision/v3/sbambach/_postdoc/marr/exp12/_", "_");
        FILE *xywh = fopen("/l/vision/v3/sbambach/_postdoc/marr/exp12/__20161022_17878/__20161022_17878_c_xywh.txt", "a");
		
        for(j = 0; j < classes; j++){
            int flag = 0;
            for(k = 0; k < side*side*l.n; ++k){
                if(probs[k][j] > 0){
                    flag = 1;
                    fprintf(xywh, "%s %d %f %f %f %f\n", imagepath, j, boxes[k].x, boxes[k].y, boxes[k].w, boxes[k].h);
                    break;
//                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
							// threshold as the proposals
                }
            }
            if (flag == 0)
                fprintf(xywh, "%s %d %f %f %f %f\n", imagepath, j, 0.0, 0.0, 0.0, 0.0);
        }
        fclose(xywh);
//        draw_detections(orig, l.side*l.side*l.n, thresh_for_prob, boxes, probs, voc_names, voc_labels, 24);//I add this to draw detections in each test image.
        // draw_detections function is in image.c
//		char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_parent/JPEGImages/", "testF/parent/");
//         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_child/JPEGImages/", "testF/child/");
//        char *imagepath = find_replace(path, "cam07_frames_p", "child_results_1");
//        imagepath = find_replace(imagepath, "cam08_frames_p", "parent_results_2");
         
//        save_image(orig, imagepath);

/*
        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);// the thresh in fact should be thresh_for_prob, 
		                                                                                           // but since the last argument is 1 and thus the code will 
																								   // return confidence score, so what number you put there 
																								   // does not matter

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);//For details about box_label and read_boxes, take a look at data.c and data.h
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
							// threshold as the proposals
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
            }
        }

        fprintf(stderr, "%5d %5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, proposals, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
		// RPs/Img: region proposals per image. The recall above means the recall of each bounding boxes because when getting the correct predictions, it only
		// consider the boxes whose confidence score and IOU are greater than their thresholds respectively as the correct predictions. 
		// In other words, it does not consider whether the program predict the right class of the object. 
*/
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

/*
void validate_yolo_recall_toy(char *cfgfile, char *weightfile)// This function is for getting precision VS recall graph
                                                              // Since probs = confidence * initially_predicted_probs, when we use a prob_thresh to 
															  // filter out some values, it in fact consider the confidence score, i.e., when confidence score
															  // is very small, it will make probs drop down to be lower than prob_thresh
                                                              // So we do not have to consider it again															  
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/test_toy.txt");
    list *plist = get_paths("data/test_toy_2.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
	box *boxes_initial = calloc(side*side*l.n, sizeof(box));
	
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
	
	float **probs_initial = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs_initial[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

//    float thresh = .001;
//    float thresh = .001;//this threshold value is for picking region proposals with a confidence score (which is exactly scale in this script) greater than the threshold
    float iou_thresh = .5;
//    float iou_thresh = .8;
//    float nms = 0;
    float nms = 0.4;
//    float thresh_for_prob = .01;//this threshold value is for picking region proposals with a probabilities of a certain class greater than the threshold
                                //since we use convert_detections_toy, we can make this value equal to 0
								
	int *total_gt = calloc(classes, sizeof(int));
    int *correct = calloc(classes, sizeof(int));
	int *correct_initial = calloc(classes, sizeof(int));
    int *proposals = calloc(classes, sizeof(int));
	int *proposals_initial = calloc(classes, sizeof(int));
    
	float *recall = calloc(classes, sizeof(float));
	float *recall_initial = calloc(classes, sizeof(float));
	float *precision = calloc(classes, sizeof(float));
	float *precision_initial = calloc(classes, sizeof(float));
	
	FILE *precision_VS_recall = fopen("/l/vision/v3/zehzhang/toy_data/Precision VS Recall.txt", "a");
	FILE *precision_VS_recall_initial = fopen("/l/vision/v3/zehzhang/toy_data/Precision VS Recall_initial.txt", "a");
	fprintf(precision_VS_recall, "Class Threshold Precision Recall\n");
	fprintf(precision_VS_recall_initial, "Class Threshold Precision Recall\n");
	fclose(precision_VS_recall);
	fclose(precision_VS_recall_initial);
	
	float thresh_for_prob = 0;
	int times = 0;
	
	for(times = 0; times < 10000; ++times){
	
	    FILE *precision_VS_recall = fopen("/l/vision/v3/zehzhang/toy_data/Precision VS Recall.txt", "a");
	    FILE *precision_VS_recall_initial = fopen("/l/vision/v3/zehzhang/toy_data/Precision VS Recall_initial.txt", "a");
	
	    thresh_for_prob = 0.0001 * times;
	
    	for(i = 0; i < classes; i++){
	    total_gt[i] = 0;
    	correct[i] = 0;
    	correct_initial[i] = 0;
    	proposals[i] = 0;
    	proposals_initial[i] = 0;
    	}
	
//    float avg_iou = 0;

        for(i = 0; i < m; ++i){
            char *path = paths[i];
            image orig = load_image_color(path, 0, 0);
            image sized = resize_image(orig, net.w, net.h);
            char *id = basecfg(path);
            float *predictions = network_predict(net, sized.data);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 1);
//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
            convert_detections_toy(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 0);
		
	    	convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs_initial, boxes_initial, 0);
		
            if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);
    		if (nms) do_nms(boxes_initial, probs_initial, side*side*l.n, 1, nms);

//        draw_detections(orig, l.side*l.side*l.n, thresh_for_prob, boxes, probs, voc_names, voc_labels, 24);//I add this to draw detections in each test image.
//        char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_parent/JPEGImages/", "testF/parent/");
//         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_child/JPEGImages/", "testF/child/");
//        save_image(orig, imagepath);


//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);

            char *labelpath = find_replace(path, "images", "labels");
            labelpath = find_replace(labelpath, "JPEGImages", "labels");
            labelpath = find_replace(labelpath, ".jpg", ".txt");
            labelpath = find_replace(labelpath, ".JPEG", ".txt");

            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);//For details about box_label and read_boxes, take a look at data.c and data.h
		
//        for(k = 0; k < side*side*l.n; ++k){
//            if(probs[k][0] > thresh){
//                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
//				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
//							// threshold as the proposals
//            }
//        }
        
            
            for(k = 0; k < side*side*l.n; ++k){
				int this_box_index = max_index(probs[k], classes);
    		    int this_box_index_initial = max_index(probs_initial[k], classes);
                if(probs[k][this_box_index] > thresh_for_prob){
    				++proposals[this_box_index];		
        		}
    			if(probs_initial[k][this_box_index_initial] > thresh_for_prob){
     				++proposals_initial[this_box_index_initial];
    			}
            }
     		
            for (j = 0; j < num_labels; ++j) {
                 ++total_gt[truth[j].id];
                 box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
                 float best_iou = 0;
				 float best_iou_initial = 0;
                 for(k = 0; k < side*side*l.n; ++k){
                     float iou = box_iou(boxes[k], t);
    				 float iou_initial = box_iou(boxes_initial[k], t);
    				 int this_box_index = max_index(probs[k], classes);
    				 int this_box_index_initial = max_index(probs_initial[k], classes);
    				 if(this_box_index == truth[j].id && probs[k][this_box_index] > thresh_for_prob && iou > best_iou){
    				 	best_iou = iou;
        			 }
    				 if(this_box_index_initial == truth[j].id && probs_initial[k][this_box_index_initial] > thresh_for_prob && iou_initial > best_iou_initial){
    				 	best_iou_initial = iou_initial;
    				 }

//                if(probs[k][0] > thresh && iou > best_iou){
//                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
//                }
                 }
//            avg_iou += best_iou;
                 if(best_iou > iou_thresh){
                     ++correct[truth[j].id];// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
                 }
				 if(best_iou_initial > iou_thresh){
				     ++correct_initial[truth[j].id];
				 }
            }
		

//        fprintf(stderr, "%5d %5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, proposals, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
		// RPs/Img: region proposals per image. The recall above means the recall of bounding boxes because when getting the correct predictions, it only
		// consider the boxes whose confidence score and IOU are greater than their thresholds respectively as the correct predictions. 
		// In other words, it does not consider whether the program predicts the right class of the object. 
             free(id);
             free_image(orig);
             free_image(sized);
		
        }
     	for( i = 0; i < classes; i++){
     	recall[i] = (float)correct[i]/total_gt[i];
    	recall_initial[i] = (float)correct_initial[i]/total_gt[i];
    	precision[i] = (float)correct[i]/proposals[i];
    	precision_initial[i] = (float)correct_initial[i]/proposals_initial[i];
		if(recall[i] > 1) recall[i] = 1;
		if(recall_initial[i] > 1) recall_initial[i] = 1;
		if(precision[i] > 1) precision[i] = 1;
		if(precision_initial[i] > 1) precision_initial[i] = 1;
	    fprintf(precision_VS_recall, "  %2d   %6f   %6f  %6f\n", i, thresh_for_prob, precision[i], recall[i]);
	    fprintf(precision_VS_recall_initial, "  %2d   %6f   %6f  %6f\n", i, thresh_for_prob, precision_initial[i], recall_initial[i]);
	    }
	    printf("Finish %d/10000\n", times);
    	fclose(precision_VS_recall);
    	fclose(precision_VS_recall_initial);
   	}
}
*/

/*
char *file_path[] = {"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c0.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c1.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c2.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c3.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c4.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c5.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c6.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c7.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c8.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c9.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c10.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c11.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c12.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c13.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c14.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c15.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c16.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c17.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c18.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c19.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c20.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c21.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c22.txt", \
"/l/vision/v3/zehzhang/toy_data/correct_proposal_total_c23.txt"};

void validate_yolo_recall_toy(char *cfgfile, char *weightfile)// This function is for getting precision VS recall graph
                                                              // Since probs = confidence * initially_predicted_probs, when we use a prob_thresh to 
															  // filter out some values, it in fact consider the confidence score, i.e., when confidence score
															  // is very small, it will make probs drop down to be lower than prob_thresh
                                                              // So we do not have to consider it again															  
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("data/test_toy.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
	box *boxes_initial = calloc(side*side*l.n, sizeof(box));
	
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
	
	float **probs_initial = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs_initial[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

//    float thresh = .001;
//    float thresh = .001;//this threshold value is for picking region proposals with a confidence score (which is exactly scale in this script) greater than the threshold
    float iou_thresh = .5;
//    float iou_thresh = .8;
//    float nms = 0;
    float nms = 0.4;
//    float thresh_for_prob = .01;//this threshold value is for picking region proposals with a probabilities of a certain class greater than the threshold
                                //since we use convert_detections_toy, we can make this value equal to 0
    
	int MAX_TIME = 10000;	
	int *total_gt = calloc(classes, sizeof(int));
    int *correct = calloc(classes, sizeof(int));
	int *correct_initial = calloc(classes, sizeof(int));
    int *proposals = calloc(classes, sizeof(int));
	int *proposals_initial = calloc(classes, sizeof(int));
	
	for(k = 0; k < classes; k++) total_gt[k] = 0;


	

	
	float thresh_for_prob = 0;
	int times = 0;
		
//    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
//     convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 1);
//     convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
        convert_detections_toy(predictions, classes, l.n, square, side, 1, 1, 0, probs, boxes, 0);
		convert_detections(predictions, classes, l.n, square, side, 1, 1, 0, probs_initial, boxes_initial, 0);//only do this once with threshold 0 to get 
			                                                                                                      //all the probabilies and box information we 
																												  //need to compute recall and precision for 
																												  //different threshold
		
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);
        if (nms) do_nms(boxes_initial, probs_initial, side*side*l.n, 1, nms);

//        draw_detections(orig, l.side*l.side*l.n, thresh_for_prob, boxes, probs, voc_names, voc_labels, 24);//I add this to draw detections in each test image.
//        char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_parent/JPEGImages/", "testF/parent/");
//         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_child/JPEGImages/", "testF/child/");
//        save_image(orig, imagepath);
        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");
        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);//For details about box_label and read_boxes, take a look at data.c and data.h
	    
		for (j = 0; j < num_labels; ++j) ++total_gt[truth[j].id];



//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        for(times = 0; times < MAX_TIME; ++times){
		    
	        for(k = 0; k < classes; k++){
	            correct[k] = 0;
		        correct_initial[k] = 0;
		        proposals[k] = 0;
		        proposals_initial[k] = 0;
	    }
	        
			thresh_for_prob = 0.0001 * times;

		
//        for(k = 0; k < side*side*l.n; ++k){
//            if(probs[k][0] > thresh){
//                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
//				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
//							// threshold as the proposals
//            }
//        }
        
            
            for(k = 0; k < side*side*l.n; ++k){
				int this_box_index = max_index(probs[k], classes);
    		    int this_box_index_initial = max_index(probs_initial[k], classes);
                if(probs[k][this_box_index] > thresh_for_prob){
    				++proposals[this_box_index];		
        		}
    			if(probs_initial[k][this_box_index_initial] > thresh_for_prob){
     				++proposals_initial[this_box_index_initial];
    			}
            }
     		
            for (j = 0; j < num_labels; ++j) {
                 box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
                 float best_iou = 0;
				 float best_iou_initial = 0;
                 for(k = 0; k < side*side*l.n; ++k){
                     float iou = box_iou(boxes[k], t);
    				 float iou_initial = box_iou(boxes_initial[k], t);
    				 int this_box_index = max_index(probs[k], classes);
    				 int this_box_index_initial = max_index(probs_initial[k], classes);
    				 if(this_box_index == truth[j].id && probs[k][this_box_index] > thresh_for_prob && iou > best_iou){
    				 	best_iou = iou;
        			 }
    				 if(this_box_index_initial == truth[j].id && probs_initial[k][this_box_index_initial] > thresh_for_prob && iou_initial > best_iou_initial){
    				 	best_iou_initial = iou_initial;
    				 }

//                if(probs[k][0] > thresh && iou > best_iou){
//                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
//                }
                 }
//            avg_iou += best_iou;
                 if(best_iou > iou_thresh){
                     ++correct[truth[j].id];// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
                 }
				 if(best_iou_initial > iou_thresh){
				     ++correct_initial[truth[j].id];
				 }
            }
		

//        fprintf(stderr, "%5d %5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, proposals, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
		// RPs/Img: region proposals per image. The recall above means the recall of bounding boxes because when getting the correct predictions, it only
		// consider the boxes whose confidence score and IOU are greater than their thresholds respectively as the correct predictions. 
		// In other words, it does not consider whether the program predicts the right class of the object. 

		
		
		for(k = 0; k < classes; k++){
		    FILE *correct_proposal_total = fopen(file_path[k], "a");
			fprintf(correct_proposal_total, "%d %d %d %d %d %d\n", i, times, correct[k], proposals[k], correct_initial[k], proposals_initial[k]);
			fclose(correct_proposal_total);
		}
		
		
        }
		free(id);
        free_image(orig);
        free_image(sized);
		printf("Finish processing %d image(s).", i);
		
   	}
	
	for(k = 0; k < classes; k++){
		    FILE *correct_proposal_total = fopen(file_path[k], "a");
			fprintf(correct_proposal_total, "%d", total_gt[k]);
			fclose(correct_proposal_total);
		}
}
*/

void validate_yolo_recall_toy(char *cfgfile, char *weightfile)// This function is for getting precision VS recall graph
                                                              // Since probs = confidence * initially_predicted_probs, when we use a prob_thresh to 
															  // filter out some values, it in fact consider the confidence score, i.e., when confidence score
															  // is very small, it will make probs drop down to be lower than prob_thresh
                                                              // So we do not have to consider it again															  
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    //list *plist = get_paths("data/test_toy.txt");
    list *plist = get_paths("data/test_toy_2.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
	box *boxes_initial = calloc(side*side*l.n, sizeof(box));
	
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
	
	float **probs_initial = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs_initial[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

//    float thresh = .001;
//    float thresh = .001;//this threshold value is for picking region proposals with a confidence score (which is exactly scale in this script) greater than the threshold
    float iou_thresh = .5;
//    float iou_thresh = .8;
//    float nms = 0;
    float nms = 0.4;
//    float thresh_for_prob = .01;//this threshold value is for picking region proposals with a probabilities of a certain class greater than the threshold
                                //since we use convert_detections_toy, we can make this value equal to 0
    
	int MAX_TIME = 10001;	
	int *total_gt = calloc(classes, sizeof(int));
    static int correct[10001][24];
	static int correct_initial[10001][24];
    static int proposals[10001][24];
	static int proposals_initial[10001][24];

	
	for(k = 0; k < classes; k++) total_gt[k] = 0;
	for(j = 0; j < MAX_TIME; j++)
	    for(k = 0; k < classes; k++){
	        correct[j][k] = 0;
		    correct_initial[j][k] = 0;
		    proposals[j][k] = 0;
		    proposals_initial[j][k] = 0;
	    }

	

	
	float thresh_for_prob = 0;
	int times = 0;
	float min_thresh = -0.0001;	
//    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
//     convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh_for_prob, probs, boxes, 1);
//     convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 0);
        convert_detections_toy(predictions, classes, l.n, square, side, 1, 1, min_thresh, probs, boxes, 0);
		convert_detections(predictions, classes, l.n, square, side, 1, 1, min_thresh, probs_initial, boxes_initial, 0);//only do this once with threshold 0 to get 
			                                                                                                      //all the probabilies and box information we 
																												  //need to compute recall and precision for 
																												  //different threshold
		
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);
        if (nms) do_nms(boxes_initial, probs_initial, side*side*l.n, 1, nms);

//        draw_detections(orig, l.side*l.side*l.n, thresh_for_prob, boxes, probs, voc_names, voc_labels, 24);//I add this to draw detections in each test image.
//        char *imagepath = find_replace(path, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_parent/JPEGImages/", "testF/parent/");
//         imagepath = find_replace(imagepath, "/l/vision/v3/zehzhang/toy_data/20150530_16859_sync_frames_child/JPEGImages/", "testF/child/");
//        save_image(orig, imagepath);
        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");
        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);//For details about box_label and read_boxes, take a look at data.c and data.h
	    for (j = 0; j < num_labels; ++j) ++total_gt[truth[j].id];


//        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        for(times = 0; times < MAX_TIME; ++times){
	        
			thresh_for_prob = 0.0001 * (times - 1);

		
//        for(k = 0; k < side*side*l.n; ++k){
//            if(probs[k][0] > thresh){
//                ++proposals;// If the confidence score is greater than a threshold, it considers this box as a proposal
//				            // Note: this is different from what we do usually, which will consider all the boxes whose probabilities are greater than a 
//							// threshold as the proposals
//            }
//        }
        
/* //Count the proposals on the whole dataset           
            for(k = 0; k < side*side*l.n; ++k){
				int this_box_index = max_index(probs[k], classes);
    		    int this_box_index_initial = max_index(probs_initial[k], classes);
                if(probs[k][this_box_index] > thresh_for_prob){
    				++proposals[times][this_box_index];		
        		}
    			if(probs_initial[k][this_box_index_initial] > thresh_for_prob){
     				++proposals_initial[times][this_box_index_initial];
    			}
            }
*/

//Count the proposals for each class separately
            for(k = 0; k < side*side*l.n; ++k){
//				int this_box_index = max_index(probs[k], classes);
//    		    int this_box_index_initial = max_index(probs_initial[k], classes);
                for(j = 0; j < classes; ++j){
                    if(probs[k][j] > thresh_for_prob){
    				    ++proposals[times][j];		
        		    }
    			    if(probs_initial[k][j] > thresh_for_prob){
     				    ++proposals_initial[times][j];
    			    }
				}
            }

/*  //Count the correctly detected instances on the whole dataset     		
            for (j = 0; j < num_labels; ++j) {
                 box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
                 float best_iou = 0;
				 float best_iou_initial = 0;
                 for(k = 0; k < side*side*l.n; ++k){
                     float iou = box_iou(boxes[k], t);
    				 float iou_initial = box_iou(boxes_initial[k], t);
    				 int this_box_index = max_index(probs[k], classes);
    				 int this_box_index_initial = max_index(probs_initial[k], classes);
    				 if(this_box_index == truth[j].id && probs[k][this_box_index] > thresh_for_prob && iou > best_iou){
    				 	best_iou = iou;
        			 }
    				 if(this_box_index_initial == truth[j].id && probs_initial[k][this_box_index_initial] > thresh_for_prob && iou_initial > best_iou_initial){
    				 	best_iou_initial = iou_initial;
    				 }

//                if(probs[k][0] > thresh && iou > best_iou){
//                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
//                }
                 }
//            avg_iou += best_iou;
                 if(best_iou > iou_thresh){
                     ++correct[times][truth[j].id];// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
                 }
				 if(best_iou_initial > iou_thresh){
				     ++correct_initial[times][truth[j].id];
				 }
            }
*/		

//Count the correctly detected instances for each class separately
            for (j = 0; j < num_labels; ++j) {
                 box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};//truth[j].id
                 float best_iou = 0;
				 float best_iou_initial = 0;
                 for(k = 0; k < side*side*l.n; ++k){
                     float iou = box_iou(boxes[k], t);
    				 float iou_initial = box_iou(boxes_initial[k], t);
//    				 int this_box_index = max_index(probs[k], classes);
//    				 int this_box_index_initial = max_index(probs_initial[k], classes);
    				 if(probs[k][truth[j].id] > thresh_for_prob && iou > best_iou){
    				 	best_iou = iou;
        			 }
    				 if(probs_initial[k][truth[j].id] > thresh_for_prob && iou_initial > best_iou_initial){
    				 	best_iou_initial = iou_initial;
    				 }

//                if(probs[k][0] > thresh && iou > best_iou){
//                    best_iou = iou;// For each object of the ground truth, it finds the bounding box with the greatest IOU and let best_iou equal to it
//                }
                 }
//            avg_iou += best_iou;
                 if(best_iou > iou_thresh){
                     ++correct[times][truth[j].id];// For each object of the ground truth, if it finds a box whose confidence score and IOU are greater than their thresholds 
				          // respectively, it will consider this as a correct prediction. However, it does not consider the probabilities of classes. 
                 }
				 if(best_iou_initial > iou_thresh){
				     ++correct_initial[times][truth[j].id];
				 }
            }

//        fprintf(stderr, "%5d %5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, proposals, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
		// RPs/Img: region proposals per image. The recall above means the recall of bounding boxes because when getting the correct predictions, it only
		// consider the boxes whose confidence score and IOU are greater than their thresholds respectively as the correct predictions. 
		// In other words, it does not consider whether the program predicts the right class of the object. 

		
        }
		printf("Finish processing %d image(s).\n", i);
		free(id);
        free_image(orig);
        free_image(sized);
   	}
	
	printf("Begin writing files!\n");
//	FILE *precision_VS_recall = fopen("/l/vision/v3/zehzhang/toy_data/___Precision VS Recall.txt", "a");
//	FILE *precision_VS_recall_initial = fopen("/l/vision/v3/zehzhang/toy_data/___Precision VS Recall_initial.txt", "a");
	FILE *precision_VS_recall = fopen("/l/vision/v3/zehzhang/toy_data/___Precision VS Recall_2.txt", "a");
	FILE *precision_VS_recall_initial = fopen("/l/vision/v3/zehzhang/toy_data/___Precision VS Recall_initial_2.txt", "a");
	fprintf(precision_VS_recall, "Class Threshold Precision Recall\n");
	fprintf(precision_VS_recall_initial, "Class Threshold Precision Recall\n");
//	fclose(precision_VS_recall);
//	fclose(precision_VS_recall_initial);
	float recall = 0;
	float recall_initial = 0;
	float precision = 0;
	float precision_initial = 0;
	for(times = 0; times < MAX_TIME; times++){
	    thresh_for_prob = (times - 1) * 0.0001;
	    for( i = 0; i < classes; i++){
     	recall = (float)correct[times][i]/total_gt[i];
    	recall_initial = (float)correct_initial[times][i]/total_gt[i];
    	precision = (float)correct[times][i]/proposals[times][i];
    	precision_initial = (float)correct_initial[times][i]/proposals_initial[times][i];
		if(recall > 1) recall = 1;
		if(recall_initial > 1) recall_initial = 1;
		if(precision > 1) precision = 1;
		if(precision_initial > 1) precision_initial = 1;//Maybe these statements are not needed?
	    fprintf(precision_VS_recall, "  %2d   %6f %6f  %6f\n", i, thresh_for_prob, precision, recall);
	    fprintf(precision_VS_recall_initial, "  %2d   %6f %6f  %6f\n", i, thresh_for_prob, precision_initial, recall_initial);
	    }
	    printf("Finish %d/10000\n", times);

	
	}
	printf("Done!\n");
	fclose(precision_VS_recall);
    fclose(precision_VS_recall_initial);
}


void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.4;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
//        convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        convert_detections_toy(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 24);
        //save_image(im, "predictions");
//        save_image(im, "testF/predictions");
        show_image(im, "predictions");

        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void run_yolo(int argc, char **argv)
{
    int i;
    //for(i = 0; i < 20; ++i){
    for(i = 0; i < 24; ++i){
        char buff[256];
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "toymap")) validate_yolo_recall_toy(cfg, weights);//Record precisions and recalls for different thresholds
    else if(0==strcmp(argv[2], "toyallframes")) validate_yolo_recall_video(cfg, weights);//Record the result images for all frames
	else if(0==strcmp(argv[2], "toyboxes")) validate_yolo_recall_coordinates(cfg, weights);//Record all the coordinates for all boxes in each image
    //else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, 20, frame_skip);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, 24, frame_skip);
}
