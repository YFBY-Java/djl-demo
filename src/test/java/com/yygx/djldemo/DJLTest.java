package com.yygx.djldemo;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class DJLTest {



    /**
     * 测试NDArray
     */
    @Test
    void testNDArray(){
        try(NDManager manager = NDManager.newBaseManager()){  // 创建NDManager对象
            NDArray ones = manager.ones(new Shape(2, 3));// 创建一个2x3的全1矩阵

            NDArray ndArray = manager.create(new float[]{1, 2, 3, 4, 5, 6}, new Shape(2, 3));// 创建一个2x3的矩阵
            System.out.println(ndArray);

            NDArray transpose = ndArray.transpose();// 转置
            System.out.println(transpose);

        }
    }


    // 完整训练一个模型
    // 1.准备数据集
    // 2.构建神经网络
    // 3.构建模型（这个模型应用上面的神经网络）
    // 4.训练模型
    // 5.保存模型

    // 6.加载模型
    // 7.预测（给模型一个新输入，让他判断这是什么）

    @Test
    @SneakyThrows
    void trainingComplete() throws Exception{
        // 准备数据集
        RandomAccessDataset trainDataset = getDataset(Dataset.Usage.TRAIN);
        RandomAccessDataset testDataset = getDataset(Dataset.Usage.TEST);
        // 构建神经网络
        Mlp mlp = new Mlp(Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH, 10, new int[]{128, 64});
        // 创建模型
        try(Model model = Model.newInstance("mlp")){
            // 设置神经网络
            model.setBlock(mlp);
            //3、训练配置
            String outputDir = "build/mlp";
            DefaultTrainingConfig config = setupTrainingConfig(outputDir);


            // 获取训练器
            try (Trainer trainer = model.newTrainer(config)) {
                // 初始化
                trainer.initialize(new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH));
                // 训练   训练次数，训练集，测试集
                EasyTrain.fit(trainer, 5, trainDataset, testDataset);
                TrainingResult trainingResult = trainer.getTrainingResult();   // 获取训练结果
                System.out.println(trainingResult);
                //保存模型
                model.save(Paths.get(outputDir), "mlp");
            }
        }

    }


    // 测试模型
    @Test
    void testModel() throws IOException, TranslateException, MalformedModelException {
        // 准备测试数据
//        Image image = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/0.png");

//        Image image = ImageFactory.getInstance().fromFile(Paths.get("build/img/img.png"));
        Image image = ImageFactory.getInstance().fromFile(Paths.get("build/img/img_1.png"));


        //加载模型
        Path modelDir = Paths.get("build/mlp");
        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH, 10, new int[]{128, 64}));  // 设置神经网络
        model.load(modelDir);  // 加载模型

        // 获取转换器
        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(28, 28))
                .addTransform(new ToTensor())
                .build();

        // 获取预测器
        Predictor<Image, Classifications> predictor = model.newPredictor(translator);

        // 预测图片分类
        Classifications classifications = predictor.predict(image);
        System.out.println(classifications);
    }




    // 获取数据集方法
    @SneakyThrows
    private RandomAccessDataset getDataset(Dataset.Usage usage) throws Exception {
        // Mnist是内置的数据集，可以直接使用，用自己的数据就自定义Dataset（数据集）
        Mnist mnist = Mnist.builder().setSampling(64, true).optUsage(usage).build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }

    private DefaultTrainingConfig setupTrainingConfig(String outputDir) {
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }







}