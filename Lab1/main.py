import numpy as np
import matplotlib.pyplot as plt


class linearClassifier:
    def __init__(self, classNumber: int = 10, epoch: int = 500, isCrossEntropy: bool = True, studyRate: np.double = 0.01) -> None:
        self.classes = classNumber
        self.epoch = epoch
        self.cross = isCrossEntropy
        self.rate = studyRate

        self.load()
        self.weight = np.random.randn(self.dim, self.classes)*0.01

    def load(self, trainData: str = "train-images.npy", trainLabel: str = "train-labels.npy", testData: str = "test-images.npy", testLabel: str = "test-labels.npy") -> None:
    # load all data
        self.trainImg = np.load(trainData)
        self.trainLabelori = np.load(trainLabel)
        self.trainLabel = self.toOneHot(self.trainLabelori)
        self.trainNum, self.dim = self.trainImg.shape
        self.testImg = np.load(testData)
        self.testLabelori = np.load(testLabel)
        self.testLabel = self.toOneHot(self.testLabelori)
        self.testNum = self.testImg.shape[0]

    def toOneHot(self, label: np.array):
    # change a label set to one-hot set
        ret = np.zeros((label.shape[0], self.classes), int)
        for i in range(0, label.shape[0]):
            ret[i, label[i]] = 1
        return ret

    def train(self, display: bool = False, drawPlot: bool = False):
    # run train for an epoch
        lossHistory = []
        for i in range(self.epoch):
            if self.cross:
                loss, grad = self.crossEntropy()
            else:
                loss, grad = self.squaredError()
            lossHistory.append(loss)
            self.weight -= self.rate * grad
            if display:
                print("iter:", i, "; loss:", loss)
        if drawPlot:
            x = np.arange(1, len(lossHistory)+1)
            pic = plt.plot(x,lossHistory)
            plt.xlabel("Train Time")
            plt.ylabel("Loss")
            if self.cross:
                plt.title("Training Result with Cross Entropy Loss Function")
            else:
                plt.title("Training Result with Square Error Loss Function")
            plt.show()
        return lossHistory

    def test(self, display: bool = False):
    # test the training result
        predictResult = np.dot(self.testImg, self.weight)
        predictResult = np.argmax(predictResult, axis=1)

        counter = 0
        for i in range(self.testNum):
            if predictResult[i] == self.testLabelori[i]:
                counter += 1

        accuracy = counter / self.testNum
        if display:
            print("accuracy:", accuracy)

        return accuracy

    def crossEntropy(self):
    # use cross-entropy as loss function
        grad_part1 = np.mat(self.softmax(np.dot(self.trainImg, self.weight)) - self.trainLabel)
        grad_part2 = np.transpose(self.trainImg)
        grad = np.dot(grad_part2, grad_part1) / self.trainNum

        loss = 0.0
        for i in range(self.trainNum):
            tmp = self.softmax(np.array([np.dot(self.trainImg[i], self.weight)]))[0, self.trainLabelori[i]]
            # print(tmp)
            # if tmp is too small, it will cause log's fault
            if abs(tmp)<1e-25:
                tmp=1e-25
            # print(tmp,np.log(tmp))
            loss -= np.log(tmp)     # minus because original equation with negative sign

        loss /= self.trainNum
        return [loss, grad]

    def softmax(self,z:np.array):
    # softmax function (without i itor)
        # minus row's max, avoiding exp(z) overflow
        rowMax = z.max(axis=1)
        rowMax = rowMax.reshape(-1, 1)
        z = z - rowMax

        nume=np.exp(z)
        deno=np.sum(nume,axis=1,keepdims=True)
        # print(deno)

        return nume/deno

    def squaredError(self):
    # use squared error as 
        grad_part0 = self.sigmoid(np.dot(self.trainImg, self.weight))
        grad_part1 = (grad_part0 - self.trainLabel)
        # grad_part1 = (grad_part0 - self.trainLabel) * grad_part0 * (1 - grad_part0)
        grad_part2 = np.transpose(self.trainImg)
        grad = np.dot(grad_part2, grad_part1) / self.trainNum

        loss = 0.0
        for i in range(self.trainNum):
            loss += np.square(self.sigmoid(np.dot(self.trainImg[i], self.weight)[self.trainLabelori[i]])-1)
        # loss /= self.trainNum
        return [loss, grad]

    def sigmoid(self,z:np.array):
        # print(z)
        ret=1+np.exp(-z)
        ret=1/ret
        return ret

# the code runs
classifier = linearClassifier(epoch=50, isCrossEntropy=True, studyRate=0.005)
lossHistory = classifier.train(display=True, drawPlot=True)
result = classifier.test(display=True)
