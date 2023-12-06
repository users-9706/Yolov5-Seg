using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
namespace Yolov5-Seg
{
    class Program
    {
        public static float sigmoid(float a)
        {
            float b = 1.0f / (1.0f + (float)Math.Exp(-a));
            return b;
        }
        public static string[] read_class_names(string path)
        {
            string[] class_names;
            List<string> str = new List<string>();
            StreamReader sr = new StreamReader(path);
            string line;
            while ((line = sr.ReadLine()) != null)
            {
                str.Add(line);
            }
            class_names = str.ToArray();
            return class_names;
        }
        static void Main(string[] args)
        {
            float conf_threshold = 0.25f;
            float nms_threshold = 0.5f;
            string model_path = "yolov5n-seg.onnx";
            string image_path = "bus.jpg";
            string[] classes_names = read_class_names("coco.names");
            Mat masked_img = new Mat();
            List<NamedOnnxValue> input_ontainer;
            List<Rect> position_boxes = new List<Rect>();
            List<int> class_ids = new List<int>();
            List<float> class_scores = new List<float>();
            List<float> confidences = new List<float>();
            List<Mat> masks = new List<Mat>();
            Tensor<float> result_tensors_det;
            Tensor<float> result_tensors_proto;
            SessionOptions options;
            InferenceSession onnx_session;
            Tensor<float> input_tensor;
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result_infer;
            DisposableNamedOnnxValue[] results_onnxvalue;
            options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.AppendExecutionProvider_CPU(0);
            onnx_session = new InferenceSession(model_path, options);
            input_ontainer = new List<NamedOnnxValue>();
            Mat image = Cv2.ImRead(image_path);
            int max_image_length = image.Cols > image.Rows ? image.Cols : image.Rows;
            Mat max_image = Mat.Zeros(new OpenCvSharp.Size(max_image_length, max_image_length), MatType.CV_8UC3);
            Rect roi = new Rect(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(max_image, roi));
            float[] det_result_array = new float[25200 * 117];
            float[] proto_result_array = new float[32 * 160 * 160];
            float[] factors = new float[4];
            factors[0] = factors[1] = (float)(max_image_length / 640.0);
            factors[2] = image.Rows;
            factors[3] = image.Cols;
            Mat image_rgb = new Mat();
            Mat resize_image = new Mat();
            Cv2.CvtColor(max_image, image_rgb, ColorConversionCodes.BGR2RGB);
            Cv2.Resize(image_rgb, resize_image, new OpenCvSharp.Size(640, 640));
            input_tensor = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
            for (int y = 0; y < resize_image.Height; y++)
            {
                for (int x = 0; x < resize_image.Width; x++)
                {
                    input_tensor[0, 0, y, x] = resize_image.At<Vec3b>(y, x)[0] / 255f;
                    input_tensor[0, 1, y, x] = resize_image.At<Vec3b>(y, x)[1] / 255f;
                    input_tensor[0, 2, y, x] = resize_image.At<Vec3b>(y, x)[2] / 255f;
                }
            }
            input_ontainer.Add(NamedOnnxValue.CreateFromTensor("images", input_tensor));
            result_infer = onnx_session.Run(input_ontainer);
            results_onnxvalue = result_infer.ToArray();
            result_tensors_det = results_onnxvalue[0].AsTensor<float>();
            result_tensors_proto = results_onnxvalue[1].AsTensor<float>();
            det_result_array = result_tensors_det.ToArray();
            proto_result_array = result_tensors_proto.ToArray();
            Mat detect_data = new Mat(25200, 117, MatType.CV_32F, det_result_array);
            Mat proto_data = new Mat(32, 25600, MatType.CV_32F, proto_result_array);
            for (int i = 0; i < detect_data.Rows; i++)
            {
                Mat conf_scores = detect_data.Row(i).ColRange(4, 5);
                float conf_value = conf_scores.Get<float>(0, 0);
                if (conf_value < conf_threshold)
                {
                    continue;
                }
                Mat classes_scores = detect_data.Row(i).ColRange(5, 85);
                Point max_classId_point, min_classId_point;
                double max_score, min_score;
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);
                if (max_score > 0.25)
                {
                    Mat mask = detect_data.Row(i).ColRange(85, 117);
                    float cx = detect_data.At<float>(i, 0);
                    float cy = detect_data.At<float>(i, 1);
                    float ow = detect_data.At<float>(i, 2);
                    float oh = detect_data.At<float>(i, 3);
                    int x = (int)((cx - 0.5 * ow) * factors[0]);
                    int y = (int)((cy - 0.5 * oh) * factors[1]);
                    int width = (int)(ow * factors[0]);
                    int height = (int)(oh * factors[1]);
                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;
                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    classes_scores.Add(max_score);
                    confidences.Add((float)max_score);
                    masks.Add(mask);
                }
            }
            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, conf_threshold, nms_threshold, out indexes);
            Mat rgb_mask = Mat.Zeros(new Size((int)factors[3], (int)factors[2]), MatType.CV_8UC3);
            Random rd = new Random();
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                Rect box = position_boxes[index];
                int box_x1 = Math.Max(0, box.X);
                int box_y1 = Math.Max(0, box.Y);
                int box_x2 = Math.Max(0, box.BottomRight.X);
                int box_y2 = Math.Max(0, box.BottomRight.Y);
                Mat original_mask = masks[index] * proto_data;
                for (int col = 0; col < original_mask.Cols; col++)
                {
                    original_mask.At<float>(0, col) = sigmoid(original_mask.At<float>(0, col));
                }
                Mat reshape_mask = original_mask.Reshape(1, 160);
                int mx1 = Math.Max(0, (int)((box_x1 / factors[0]) * 0.25));
                int mx2 = Math.Max(0, (int)((box_x2 / factors[0]) * 0.25));
                int my1 = Math.Max(0, (int)((box_y1 / factors[1]) * 0.25));
                int my2 = Math.Max(0, (int)((box_y2 / factors[1]) * 0.25));
                Mat mask_roi = new Mat(reshape_mask, new OpenCvSharp.Range(my1, my2), new OpenCvSharp.Range(mx1, mx2));
                Mat actual_maskm = new Mat();
                Cv2.Resize(mask_roi, actual_maskm, new Size(box_x2 - box_x1, box_y2 - box_y1));
                for (int r = 0; r < actual_maskm.Rows; r++)
                {
                    for (int c = 0; c < actual_maskm.Cols; c++)
                    {
                        float pv = actual_maskm.At<float>(r, c);
                        if (pv > 0.5)
                        {
                            actual_maskm.At<float>(r, c) = 1.0f;
                        }
                        else
                        {
                            actual_maskm.At<float>(r, c) = 0.0f;
                        }
                    }
                }
                Mat bin_mask = new Mat();
                actual_maskm = actual_maskm * 200;
                actual_maskm.ConvertTo(bin_mask, MatType.CV_8UC1);
                if ((box_y1 + bin_mask.Rows) >= factors[2])
                {
                    box_y2 = (int)factors[2] - 1;
                }
                if ((box_x1 + bin_mask.Cols) >= factors[3])
                {
                    box_x2 = (int)factors[3] - 1;
                }
                Mat mask = Mat.Zeros(new Size((int)factors[3], (int)factors[2]), MatType.CV_8UC1);
                bin_mask = new Mat(bin_mask, new OpenCvSharp.Range(0, box_y2 - box_y1), new OpenCvSharp.Range(0, box_x2 - box_x1));
                Rect rois = new Rect(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                bin_mask.CopyTo(new Mat(mask, rois));
                Cv2.Add(rgb_mask, new Scalar(rd.Next(0, 255), rd.Next(0, 255), rd.Next(0, 255)), rgb_mask, mask);
                Cv2.Rectangle(image, position_boxes[index], new Scalar(0, 0, 255), 2, LineTypes.Link8);
                Cv2.Rectangle(image, new Point(position_boxes[index].TopLeft.X, position_boxes[index].TopLeft.Y - 20),
                    new Point(position_boxes[index].BottomRight.X, position_boxes[index].TopLeft.Y), new Scalar(0, 255, 255), -1);
                Cv2.PutText(image, classes_names[class_ids[index]] + "-" + confidences[index].ToString("0.00"),
                    new Point(position_boxes[index].X, position_boxes[index].Y - 10),
                    HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 0, 0), 1);
                Cv2.AddWeighted(image, 0.5, rgb_mask.Clone(), 0.5, 0, masked_img);
            }
            Cv2.ImShow("Result", masked_img);
            Cv2.WaitKey(5000);
        }
    }
}