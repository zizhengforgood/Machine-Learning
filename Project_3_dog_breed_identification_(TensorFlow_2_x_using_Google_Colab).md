# Multi-class Dog Breed Classification

This notebook builds an end-to-end multi-class image classifier using TensorFlow 2.0 and TensorFlow hub.

## 1. Problem

Identifying the breed of a dog given an image of a dog.

When I'm sitting at the cafe and I take a photo of a dog, I want to know what breed of dot it is.

## 2. Data

The data we're using is from Kaggle's dog breed identification.

https://www.kaggle.com/c/dog-breed-identification/data

## 3. Evaluation

The evaluation is a file with prediction probabilities for each dog breed of each test image.

https://www.kaggle.com/c/dog-breed-identification/overview/evaluation

## 4. Features

Some information about the data:
* We're dealing with images (unstructured data) so it's probably best use deep learning/transfer learning.
* There are 120 breeds of dogs (meaning there are 120 different classes).
* There are around 10,000+ images in the training set (with labels).
* There are around 10,000+ images in the test set (without labels).


```python
#!unzip "/content/drive/MyDrive/Colab Notebooks/Dog Vision/dog-breed-identification.zip" -d "/content/drive/MyDrive/Colab Notebooks/Dog Vision/"
```

### Get the workspace ready

* Import TensorFlow
* Import TensorFlow Hub
* Make sure GPU


```python
# import necessary tools
import tensorflow as tf
import tensorflow_hub as hub
print("TF version:", tf.__version__)
print("TF Hub version:" ,hub.__version__)

# check for GPU availability
print("GPU","available(YES!!!)" if tf.config.list_physical_devices("GPU") else "Not available")
```

    TF version: 2.4.1
    TF Hub version: 0.11.0
    GPU available(YES!!!)


## Getting the data ready (turning into Tensors)

With all machine learning models, the data has to be in numerical format. So we'll turn images into Tensors (numerical representations).


```python
# checkout the labels of data
import pandas as pd
labels_csv=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Dog Vision/labels.csv")
labels_csv.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>breed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10222</td>
      <td>10222</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10222</td>
      <td>120</td>
    </tr>
    <tr>
      <th>top</th>
      <td>8b7b0f3b6474962448c419ed8c46712a</td>
      <td>scottish_deerhound</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>126</td>
    </tr>
  </tbody>
</table>
</div>




```python
labels_csv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>breed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000bec180eb18c7604dcecc8fe0dba07</td>
      <td>boston_bull</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001513dfcb2ffafc82cccf4d8bbaba97</td>
      <td>dingo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001cdf01b096e06d78e9e5112d419397</td>
      <td>pekinese</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00214f311d5d2247d5dfe4fe24b2303d</td>
      <td>bluetick</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0021f9ceb3235effd7fcde7f7538ed62</td>
      <td>golden_retriever</td>
    </tr>
  </tbody>
</table>
</div>




```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
# how many images of each breed
labels_csv['breed'].value_counts()
```




    scottish_deerhound      126
    maltese_dog             117
    afghan_hound            116
    entlebucher             115
    bernese_mountain_dog    114
                           ... 
    komondor                 67
    golden_retriever         67
    brabancon_griffon        67
    briard                   66
    eskimo_dog               66
    Name: breed, Length: 120, dtype: int64




```python
labels_csv['breed'].value_counts().plot.bar(figsize=(20,10));
```


    
![png](output_9_0.png)
    



```python
labels_csv['breed'].value_counts().median()
```




    82.0




```python
# view an image
from IPython.display import Image
Image("/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/000bec180eb18c7604dcecc8fe0dba07.jpg")
```




    
![jpeg](output_11_0.jpg)
    



### Getting images and their labels


```python
labels_csv.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>breed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000bec180eb18c7604dcecc8fe0dba07</td>
      <td>boston_bull</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001513dfcb2ffafc82cccf4d8bbaba97</td>
      <td>dingo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001cdf01b096e06d78e9e5112d419397</td>
      <td>pekinese</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00214f311d5d2247d5dfe4fe24b2303d</td>
      <td>bluetick</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0021f9ceb3235effd7fcde7f7538ed62</td>
      <td>golden_retriever</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create pathnames from iamge ID's
filenames=["/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/"+fname+'.jpg' for fname in labels_csv['id']]
filenames[:10]
```




    ['/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/000bec180eb18c7604dcecc8fe0dba07.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001cdf01b096e06d78e9e5112d419397.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00214f311d5d2247d5dfe4fe24b2303d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0021f9ceb3235effd7fcde7f7538ed62.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002211c81b498ef88e1b40b9abf84e1d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00290d3e1fdd27226ba27a8ce248ce85.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002a283a315af96eaea0e28e7163b21b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/003df8b8a8b05244b1d920bb6cf451f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0042188c895a2f14ef64a918ed9c7b64.jpg']




```python
# check whether number of filenames matches number of actual image files
import os
if len(os.listdir('/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/'))==len(filenames):
  print("filenames match actual amount of files")
else:
  print("filenames do not match actual amound of files")
```

    filenames match actual amount of files



```python
os.listdir('/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/')
```




    ['e24af0affe6c7a51b3e8ed9c30b090b7.jpg',
     'e81f7ae7181b7efe18eb433b0079bdc0.jpg',
     'e6e89b7a7579de5c5ced94253491953e.jpg',
     'e0ba27b6d9250ab716da71da96b3bca0.jpg',
     'df12a66b9b154c8bbe922846944e2ef2.jpg',
     'dcfa82b89aa735341c1d5dc52f331b89.jpg',
     'e5c20fbe4702370b3dc2690bb72a8cd7.jpg',
     'e03cdc6533bf441f8bdb7467b1039996.jpg',
     'e89f2129dc5830c7ba5114c7c79ee459.jpg',
     'e4245709e4060e08146b5fe1af72385d.jpg',
     'dbde111456535431c6caedfa5b8c8dea.jpg',
     'df2b16112524fe0c873530ac2357368c.jpg',
     'e6689ca3adab4609b12a24868b8d3bef.jpg',
     'dc068bd9527c4b3d7cf62fa73ae23bbe.jpg',
     'dcda81d6b22e37e4c0fc36a383a61e73.jpg',
     'e0acf08c678ede145c9824dbb7c3718e.jpg',
     'de14c220f31820f5c9e13ee876d95a8a.jpg',
     'e9192e92049f545405dfbb8a2a05a8c7.jpg',
     'e146cabda18bbb2cb402dece1a7dd4ef.jpg',
     'e07e46622e95d95ed698113415aaeda6.jpg',
     'e4cb3bc8fc0dd15216e26c00a7b17d81.jpg',
     'e07277bc6a6a0da06598b3ad43a011b2.jpg',
     'e8d0712b9ca47f77b333858183a38ef7.jpg',
     'dd48373a8d9f30daf5f7b51a3beb1325.jpg',
     'dba4ae5fe8b6c1651111ce96c6fbacb4.jpg',
     'e278e285726d3ec4b737d80cc891d266.jpg',
     'ddbc4983d977bc4e0f4e622da3dd747b.jpg',
     'e6898f5cc43bb4a30970f2d2210fdb9a.jpg',
     'e623bb54da50bddad89dbe36582ef530.jpg',
     'e517cb6c5a2942b6a1b545e84551372f.jpg',
     'e153e4725d7323d1833817d7cc1c9b01.jpg',
     'e1cacd82bce4a66313debf5e883fa806.jpg',
     'dcf5b9d3ab80c6829b10d20b72b67746.jpg',
     'dc19a7f22b2f78763083633f21db6602.jpg',
     'dd19ba0779065dbab0c28842fdf5851f.jpg',
     'e15dfa0f3c2c47fec01f048312ca8bb7.jpg',
     'e1fa947ecb1ca186470952a971376bc6.jpg',
     'e5eed37393749b2aabc60aef5cb0cf27.jpg',
     'df08c075e337bb17c6dd66096f3dff4f.jpg',
     'e76687c549303fb57aae9866739f44b8.jpg',
     'dd694fe7f5b46a6dd7c545d31004e1f6.jpg',
     'e4bb015a94aec342233058e7bf430704.jpg',
     'e270622b5ffec8294d7e7628c4ff6c1e.jpg',
     'dd826e6990638acaff24142f0d4b0b4b.jpg',
     'e58621674644d3729585675ecf09c091.jpg',
     'e67eb5ac576a272a62033127bf4e6570.jpg',
     'e485830e23ef663878896f45f85751f4.jpg',
     'e58763e5a8e37bcede344260eb20f26d.jpg',
     'e212cce60c302d24e8aaf0fb87ee0b2c.jpg',
     'dc226c922a709fd866d25d7d93b859c9.jpg',
     'e0d0ab2c3c2a2c157c34889ca09e2dd7.jpg',
     'e520916265354a8ba4b27fb074795c6a.jpg',
     'e6b47a2f62d0956fae1f3970f310b64a.jpg',
     'ddf23184178185362a7f99031328aae8.jpg',
     'e59d5676c3b4910024b164265ace3217.jpg',
     'e167b2451fa1a46c231f4a628800b5db.jpg',
     'e79f822b26adace4455db1fcd94ab47e.jpg',
     'e94234a15721850c8aaf3a94d2363f4f.jpg',
     'e2bbe6eedea8520efe8c709d091be39a.jpg',
     'e1f5774f729e887a7314dfaa8fcb0b8a.jpg',
     'e507c47a34012fb435804f89797b6abb.jpg',
     'e05db2e9a15a9af373794a95efee44df.jpg',
     'dd52583a6a9bfdcc5278c5d61a57b7e1.jpg',
     'e3b0ae3127ce33aecce9f512db0051e1.jpg',
     'e4a669bdb348065f7760d43fa3efa731.jpg',
     'e0032d2cfc32a75f448d9f931f6bca19.jpg',
     'dcc19db6b4e117d5e4f7d1b00fbe3982.jpg',
     'e2399ad3385db144d54b63986e8247f1.jpg',
     'e827e6439b32ea68d51e894a7350c699.jpg',
     'e63ce7912e8a260513f3d4d9db3eaa62.jpg',
     'dd1306492b2087d880af68186b145604.jpg',
     'e614058019fbb49489ee4f9090cf25ef.jpg',
     'de578be532caa0a0cfecd1a553b7e373.jpg',
     'e87e709987ed707393533e9d2429f585.jpg',
     'e8e9ea1506e0ad9382665b2690ecb4e9.jpg',
     'dd2026cafded2318431c3c78f5026fc9.jpg',
     'dee927ec2ed2efb5ea8bda227d0be2b5.jpg',
     'decaf524b73836c58c7eb371cce2b980.jpg',
     'e342333b4dd5d53c4720bb69b72cfe4e.jpg',
     'e6817d4d0b53ec2dc5339166bcb167c0.jpg',
     'e50655050c46e29c5d61527f391475b4.jpg',
     'e4ca855e09eabf518ed39a0a545da981.jpg',
     'dbe9a2868d227a12197fb6c44b93d85a.jpg',
     'e85f6d7af1405365c8e576d0a0aa7a2b.jpg',
     'f60ad1508afefc6ce94cd896ff1cdb3f.jpg',
     'ea1c23bc0a6d8c057a12d2a68b44c7c2.jpg',
     'f6082c4fcbaec253ffe964f13aba012d.jpg',
     'f55d584110183b570d7b9d03e355a403.jpg',
     'efe80b984a40901f78132386e1ac3c4b.jpg',
     'eea74111b12a1cc83201158a72cc141a.jpg',
     'ea8a9c29c83b5f3d4b917922a0ce9066.jpg',
     'f6eed8d840086c27a17d7bc4f276dd56.jpg',
     'f09536e93c84e70c1a6e144452ed98fb.jpg',
     'f316697874ee29b9ee1897979a58e90c.jpg',
     'ece252ae0a9e27590c6f30936f4166f6.jpg',
     'ec8ae6ea9408c47092ca95031ebc6acd.jpg',
     'ef50a04a3363a88a9ab1c8e5f2bfd687.jpg',
     'f24f230f4a39f5411546ab1da48cc196.jpg',
     'f065f6e63169c7d2ec46174000a2d87e.jpg',
     'f2ef034a335cdd15cda34408d3e89998.jpg',
     'ef1129cb127bff04fd3940d94f2a259b.jpg',
     'f41b28e57b9838e049dcaa3e3f11d9c9.jpg',
     'ebfc57d72a4167ec2f738f2adec52e31.jpg',
     'f33c92eca07a03a242b4a9986fc66d9b.jpg',
     'ea113a236df0ae1d6875259ed2d2bcde.jpg',
     'e9ac785eb7c5e7c31dd54be18e40bff2.jpg',
     'eca1b1928e00e376ac39717ea933b57b.jpg',
     'eadb9e961a7835d4da014f7de12e7660.jpg',
     'eeca7236c506c6c3e4405c959944f3c1.jpg',
     'ea42de31f9b6e2c8f44a48ab53bc5fd4.jpg',
     'efd5885fbb8243c327c98eb9505742bd.jpg',
     'ef4efe7980d873475501b56804670c6a.jpg',
     'f6209cf3313e3368c376dbfa13f335cc.jpg',
     'f59c6ac65a3b996404ccce16dad310e0.jpg',
     'f5cef8edf2725dc143ead4774ddff74f.jpg',
     'f194e48d99e5e0f279eb8cd5e6872e36.jpg',
     'f3c1bb36d86a8fd1436bf3ff219c8686.jpg',
     'ee88ac6bfed3abb8713bbc988b87eb80.jpg',
     'ed8a0ac4b8566dec190f7308933a1d62.jpg',
     'ee6b47bfe941098216982c8e42f2912b.jpg',
     'f5ced876dc38fce8230cb1c721e29973.jpg',
     'f67c86c7429bfe3a576c2468d243b29f.jpg',
     'eff6f9f023d63b83056ee3cabd204d9a.jpg',
     'eda5b27feb982dce425c1731a1174e1a.jpg',
     'f5bc69deded716bae5e2134c81d9a606.jpg',
     'ed89cbbf3f0c9383a7514962cd263aa9.jpg',
     'ea0d557dd0acd9191f45e703d18be9b3.jpg',
     'ec3445c0c4db2d219b2377cb0eb8f3dd.jpg',
     'f312a698c57b976f011a6d54607e81b7.jpg',
     'f48432fa098e002cc0d21d1258633697.jpg',
     'ec3985e4104dbdb2b4628ed9379d1a56.jpg',
     'e9b8e25755fcc201168fdf2d299e5350.jpg',
     'eb5feb2ca80e43548e0cc0d0b4232bd8.jpg',
     'f2dd5a812a7126c65887f87b7d0c0601.jpg',
     'f4e8482c69e4e562d86149b45da8565e.jpg',
     'ee3ffc332086a4e74209a97b7b1330da.jpg',
     'f679cd9c45865bd983920f79a2d85de3.jpg',
     'f1e82eac53a1fc3f2e3e4b151537c5ed.jpg',
     'edcf20f5ebcab09f0db88d0358b3ab56.jpg',
     'f33b851311543ba4197c7bcb0ea5757c.jpg',
     'eb921996f4cc09c35970d0c18a415a2f.jpg',
     'f5d05878affab9747f86aa1f13c52bd2.jpg',
     'f61250c6e9263857befe7f256f9e909a.jpg',
     'f46f1e81fe9224c6ee6206ff24aa6b31.jpg',
     'ece92625596397f614310352584d7b74.jpg',
     'eb83b7e5d8515928cc6990d019cfdac7.jpg',
     'f6ba94d33566242b8f4abdfcbd8a6bcb.jpg',
     'f31950a05e0c02c69042b6d3bcb3682d.jpg',
     'ecefbe28f3b43d4394e6052802bcd042.jpg',
     'ec1f2e671f8b4531b203642664173ffd.jpg',
     'ea77bbe28a68e90a21bbef677493ad65.jpg',
     'f41a4709ea45d334e7fd4f61f4ca01c0.jpg',
     'f3f01549c30b4d1ae3e6701b729ab080.jpg',
     'f5aae143adf81f97da05095d78d73350.jpg',
     'ecca49fac07d1511d00f2e8a669b1b09.jpg',
     'f28efce22982ed7867a67df7d0762c58.jpg',
     'f069ed4df76d788ff43403f1ba4fa4a8.jpg',
     'f3afd4f69038b249042cd481f0b5813a.jpg',
     'f3cf90e3a8a78d31850eca467122183a.jpg',
     'f2d5c11258efb9e8de45a81a6982f8f2.jpg',
     'ebb32c71dd92a3ad87d16006507fe779.jpg',
     'eb0dd89eae4856af66b34d736e66849c.jpg',
     'f1f03709a6fb351e6255cabc277c5fc4.jpg',
     'eba058daca887257dc71bfd2a0267d29.jpg',
     'f37bcde663c39b566fb9ce69c074a7e8.jpg',
     'f06d9112ed14edb4038daa54f1b1abd5.jpg',
     'eaa305d6f31664b513c89946b5193ca4.jpg',
     'f66c888ffa1ff06fcc2eddafb3e13aaf.jpg',
     'ebf9a3eab9607d1ef1aaaec90de886c9.jpg',
     'f63ccfe3730fabe9537060a5e9ccbd24.jpg',
     'f5379f2502e90983e6361d4acdf0a56e.jpg',
     'f3e5d0f516c56df7b58b4488f13bb00f.jpg',
     'f5091fcbb9e639158ad406a3d793ff62.jpg',
     'f5c21557b2fdf621b48720357ea35df2.jpg',
     'ed563a008568fe9ca11f9dc662826c0c.jpg',
     'f591763666758b806f71459819ce8602.jpg',
     'ed894588cade79d0253082f96f2a6310.jpg',
     'f36a3d05e27b5e206ac89b4a7e133e57.jpg',
     'f25a4184e33741e745175fd1d7c6d172.jpg',
     'efb995a30e2672d00cd70273e911590f.jpg',
     'f28df9c49c705281298bc288ffab24b2.jpg',
     'ebc1adf5291f968954104e7d2c7e540a.jpg',
     'eeef7ff1d9b702cc5eb3befca986f723.jpg',
     'f2ab1692e3ef27cffb8cbb7ad6d99092.jpg',
     'f6c308322ba8e80c9ad20f9b292a8c4c.jpg',
     'f2ad1868e6784b428fdccffe0314867f.jpg',
     'ed1e8f8f6231a658f173cc2e4c74af86.jpg',
     'eb44fc73de4f0d0f247c84284ae3c70d.jpg',
     'f2f49abc5f9f5a0f1f5d25807e76d909.jpg',
     'f2215c687883ed200475bc1bdc448622.jpg',
     'ee8ed8bcfdf48a835e39bb8deca77249.jpg',
     'ed5920d873da0891d86d9f3a25f2c2fa.jpg',
     'f19b045bcf373d6fcf05c80247850d50.jpg',
     'eebf13123af2e0241f64d35f8cddea86.jpg',
     'f3c645a41e15598e326c368253d0703f.jpg',
     'f58531d21504bc086aa5d0e2c945f988.jpg',
     'f4aa5ed5ebff85d0f8b4d1ed111831df.jpg',
     'f485a9d5afe66b74092e8178f67d34c0.jpg',
     'f019e943f7b7267ecd2665eeea73a4ce.jpg',
     'f1b4c01dfbe424368a7d9404e1f0f6c3.jpg',
     'ec1654e55864cef34ec43800744f8ac2.jpg',
     'f59d3ddb3652e870bea7cb3b15894708.jpg',
     'eb62cd1de3f75c74bac9941e7f6e8a19.jpg',
     'efec2f66576df7aa036d77ad1fafc8bf.jpg',
     'efeeefcc693ae837060c1c103195c4a6.jpg',
     'f662df0beaf91c254157863bf00ac62e.jpg',
     'f0ede8a20fcaded1594e274d98670e3b.jpg',
     'ef608e79d3e91f00896e2584c66a1709.jpg',
     'f35fd9d0193122c919f159e41769a4ca.jpg',
     'ea06fc45fe1d39398d6ad82c2f741742.jpg',
     'f6d25f988a2c379fe7c7ffc52fc39035.jpg',
     'eecbc546943281453c0e598587ec674a.jpg',
     'f09b337e8e9424b208854b361f28e39e.jpg',
     'e9f77a21c65963a5e82079536a05c811.jpg',
     'ec0f39cdcc3647246f44ac890a39a7a7.jpg',
     'f4a331375dc879c5bfc682221d143fb0.jpg',
     'f07d7513b336f4cf1ac5be3b46eecd66.jpg',
     'ea12edc949e2572580fa237d3daea938.jpg',
     'ee64bb93cf0e12fc0549b89083f9347b.jpg',
     'f2d2443edb0bffec560ed1339613503d.jpg',
     'ea18d9dba21ed0af962a29cbca958bb8.jpg',
     'f14e83eb54dc159f17ec0a676214a514.jpg',
     'edf5071a8b480af1212710874ad2c05e.jpg',
     'eb7beb7875b92b0d8861826a420c8844.jpg',
     'f5d4d250dfe2f60b155ec2f40df73936.jpg',
     'ea5c48cc2c2d348add796a1530eb27f5.jpg',
     'ea5be84db93c42752547df879813ce47.jpg',
     'f2caac437e0ad55c130321aac6ba375b.jpg',
     'f2b4247f7397d329557b0356b5645e70.jpg',
     'ec34bb628f628ac8f0c5cd9cef6dc391.jpg',
     'ebc4076f8944a2451771bc2b84859dc4.jpg',
     'f1a9a63f0ec70494ec2050de0fd39402.jpg',
     'ea426774d331299b8e12f53955e0fe99.jpg',
     'f056299c0c659aa73947f76837369a9f.jpg',
     'f20709a67b3a33528b56cfbb4ef3237b.jpg',
     'f41a7b3079bbeda29028c5919e0d84af.jpg',
     'f471dbea5a623c425c8d14b885616ba0.jpg',
     'edc923ef3b9724d44481d959573b3afd.jpg',
     'f2fb3f693de68e2564fbff2890f7ad40.jpg',
     'ee6712eed8607d6d5efd13276dcd1056.jpg',
     'eca90f594e91b6689c0908e365cac5ce.jpg',
     'eda47249346bbb28d89a524e3ff0c66b.jpg',
     'eff315e6d8b855fc2166ab4fc2a192d5.jpg',
     'f5bf73d1e0ef05b8c5fba19d051fc9f1.jpg',
     'f6c6e52ad7508381c9b545e7e4f77d8d.jpg',
     'f3973f4c9756bbad4b3b6d9ea459a15d.jpg',
     'edc0b13c478a6e50e9e8c3acd181d885.jpg',
     'ed8bd115216fc760fd0ae45471791f7f.jpg',
     'eb40629be9b4a1676e2277d19c43e4b0.jpg',
     'f5dcfc042f10a6a57a54d42a4a1f2406.jpg',
     'ecaec07d20ed3a12fc32314cf83c9370.jpg',
     'ecc1a9dcc73f00726ce6a7e79ef10a1c.jpg',
     'ee6bc7f8018f819ec1a458ea10be820e.jpg',
     'f610363215fd1f00edef4bbf9b3caba9.jpg',
     'ecec3a80a97ce33d7c6a25c96421a26d.jpg',
     'f6505af99611986f0141aa986e71e766.jpg',
     'f0216b542b2feaf7fea3412ce949536e.jpg',
     'f597a565d8b2e0d4ff1717a7d042472f.jpg',
     'edeb63f8216445c1fd9572929e5e2557.jpg',
     'f12ffab8a865e6b4d8afcb0537a746ad.jpg',
     'f0e2319cf6cf322ea6f5849eb4102fac.jpg',
     'eaa3b3bff09bb80c64937996a1be2d92.jpg',
     'f42afcc1fabac064a07c6a5b7ae2766e.jpg',
     'ed25d380c4d647930c2d9322c78a05ea.jpg',
     'ed46eed049a4fc710697517359b5a95c.jpg',
     'f690468c2bf0020c6e4ee135278cf71c.jpg',
     'eb14b97d9dc86f30f3e12d8c6da185d7.jpg',
     'f079fbbb8684c81747227bef5303c56a.jpg',
     'eda5e03fc28a7fd651cd1a7b864ae749.jpg',
     'edaba2592430f96c84ef0ae0817c4328.jpg',
     'ed756805b14606580f9740ba00feb0fe.jpg',
     'f3784274b0248ee82c3eba0ef02f5191.jpg',
     'f2f05d15bf695d5bb45ca909887230f9.jpg',
     'ef3dc6a3af2210c9cc59e2bbdc9e9ae1.jpg',
     'ef8639d4717248c8db31398c26580537.jpg',
     'ec483170d4a9c12f9f7bd0d691de7c6d.jpg',
     'f6b08ab15d3448bcfa29aa59a552a742.jpg',
     'ea9ae902f4ca9241c1187992c51db9bb.jpg',
     'f08adfc00c474e9296e74e8cc0c3bf5c.jpg',
     'f16a87262b19602f017d48f050aed0de.jpg',
     'f411d5b28c9174da6ec9c0a85837fbfb.jpg',
     'ef68f5e897135b01ea02853efd8e8344.jpg',
     'ed154eec8a5e855fabc7c6e37edbac1c.jpg',
     'f62145318e69a5061563eca2eb806d6e.jpg',
     'ea57f2db5aca0955ff2eb8ba7ea8acc2.jpg',
     'f2039c11f872cbd5e01c2bbb959f7d45.jpg',
     'f3972dd0068dd6e3b9b2da13c71c5af3.jpg',
     'f34628fdce6eb2c69ad9c65d0f44f2f7.jpg',
     'ef761ccf005b7bb9bd6a094df7e07df5.jpg',
     'eb3d4ad6a8883206ec91271f983350fd.jpg',
     'ebdf65f440e3f572374c3fd8a0e6ad08.jpg',
     'f644ec58aab5f57925bd55969f4b925b.jpg',
     'f692fcf95607b6e82528b0e90d705725.jpg',
     'f38dc765277ae9dc0bcc677dd8685c3c.jpg',
     'ec981093099a693748c575e418e7a0d2.jpg',
     'f6652a6cf81ed972d896b7c51e6aa39f.jpg',
     'f4f070765892e18ce99343779e9058cb.jpg',
     'f0649324ca9ee431d8c37ef361a0d9fd.jpg',
     'ea43ab0fec05595317c008853e9798ef.jpg',
     'f40a0d368a62d9e2d2fb2d7583368538.jpg',
     'ec02eb6e9f6814f7d0f9dda0f642260a.jpg',
     'f589e77c9b0facc8ece5f07e3c04fc46.jpg',
     'f62f6d147bb05a257c05ffe418ae4e45.jpg',
     'ecb279d5c137b233683c6e0b8d7c1b88.jpg',
     'f5047cc73f0f6691dc0e08c93ebe26da.jpg',
     'ef0baf24724a1e0f87543909d5c7dafb.jpg',
     'eb24ed36b0bc25db7e64a22763b3d20d.jpg',
     'f0dafbc65d9ddbb847863d8d510b3948.jpg',
     'f5b981b89e40c702cbaaedb8b9fc7739.jpg',
     'f27909f3037a7b5443193d7101277fb6.jpg',
     'eee79170e91e4fc80e2faa8b0d4321ee.jpg',
     'edb50bbbbf53fb184cbc04f21dd97b81.jpg',
     'ed432c00b3109a6ce7abce09d98be1e2.jpg',
     'ebc378abeb610e6d4f2a8ad0d731ccfb.jpg',
     'ea81932e294837391a437fd513a87e89.jpg',
     'f293c3e2fcc58e1d539aa71c56be8d55.jpg',
     'f550e44352e4eeb99932441678a8af34.jpg',
     'ead4174291a75b3567c459ec847c87e9.jpg',
     'ea1a1b02e734d1130c80e45880c106d3.jpg',
     'f04043819ed833d9257fa14f4d39c91f.jpg',
     'ed8fd9eef589a6fe4a0e74bf7ea77bd6.jpg',
     'f02c7f7b30bf8535b9feed3accacccb4.jpg',
     'eb64bbda1cedd44da3a37c65e4eacb9a.jpg',
     'eecab13ff6ae86429d808115d161d455.jpg',
     'f340906abda537fa75a9b73ba2742a88.jpg',
     'ed9c22ad21413691f548f5d72e7a76d9.jpg',
     'f69495e6688f3e331ecddb474367e923.jpg',
     'f4c570b8d49c119bbb2e988b240c95a4.jpg',
     'f3d98aab47dfabd6c98750f45f6c4038.jpg',
     'f58364e9a181f9d7e149f4cec3e9ec80.jpg',
     'efe33a3e4c939748548a052f5f6d83bd.jpg',
     'eda44fc4fded6ff594aeb833634ea44c.jpg',
     'eff87e94b7ba6bc2c8ddfc5e135c1892.jpg',
     'eafb5a5a9adb3e595eba451f571bfd3d.jpg',
     'ea7a7a80860e733c670e387bfe93bf08.jpg',
     'ecf239b27efded2e309b147ebd5a032d.jpg',
     'f39409a147cf719a7e34712034b41625.jpg',
     'ebefff12595565753410459216e604a9.jpg',
     'f4520fd1cb3d76b7d36acdb63dd964ca.jpg',
     'e9c694db39c5b8c904bf36c385fb171b.jpg',
     'ea15f35659972aef8ce3472d65037c4d.jpg',
     'ea829d433019aedcdeb0215c10ea905b.jpg',
     'ec6b36707749098ea596f0e97a33f0c9.jpg',
     'f3aa61d28f81531b81f15ee65d91ecec.jpg',
     'f1a94cfd3e3c67736b961587e1795a1a.jpg',
     'f5fa365252241c8be7b95d04444d5bed.jpg',
     'f3c31d575505827557a5fe80f7bba070.jpg',
     'f4045ccfa988393752a265e086979e6c.jpg',
     'ec3180f25c4860682350127a1a0c3c4a.jpg',
     'eb68d9469a3925897dd1c06bbc5a40d9.jpg',
     'ee8315e3cef238ea380d197ceb26a476.jpg',
     'e997eaa38f75d7660fd6c488e3c76d72.jpg',
     'efbabde6fc97bb48c8c8b6b75bfaea59.jpg',
     'f2580ac6cf1bb3317c661133fc3bc7b6.jpg',
     'ef80922725c7fb9865005222ef30714e.jpg',
     'f056ee6bc913fba2697dabf37c1c8531.jpg',
     'f5940c13d959f10561fda7afad8510c2.jpg',
     'ed28715e3609b8bc3674a2191310eed1.jpg',
     'e9f19fbf059a074a0b40c5b088f4fa6f.jpg',
     'ea9134791fb20bce0b72caabfa0947a6.jpg',
     'eab1276af0a54dfdf9d0311916ea813d.jpg',
     'f523e20f7fac8e1d9909d0956a468467.jpg',
     'e9a3813892432cb6d5c771156b4bb3c5.jpg',
     'f56093b4b1f055c76701c12620ace957.jpg',
     'f13e0ce50756b5eb78404b8ee5626c87.jpg',
     'ef519fe81769185abaf3bf9615f31b02.jpg',
     'f6575543e0456836d1fc6871e586a6e0.jpg',
     'eac45b011f55db4480995fb0643c54d2.jpg',
     'f3e1cc6183fe3457dd644320d730b3a4.jpg',
     'f27b4323bae39abf810bcd145d8de276.jpg',
     'f2fd565d275c740f2f4dd91f0759b9df.jpg',
     'f48ebf5f79e746455810091b884e8eb8.jpg',
     'f5b9b43b95fbf49626ede41a02f6cf1c.jpg',
     'e9fc775bd40d6d7273ff093fa12a0574.jpg',
     'ec23516f1da6f2fa32048c0920a8ef7e.jpg',
     'f3831b2287b3858b44885ec97b37c2c5.jpg',
     'ec3ea54e26c17014faf12c362f9463d5.jpg',
     'ef91df1385e44b81c78dd345b1a17f14.jpg',
     'ee92839020f1795e7eab77358e2528f1.jpg',
     'ee3e8a3091a0171973d8a9ae0faa5d28.jpg',
     'ee5eb948999e9f9de1ee4497faa0ffff.jpg',
     'f61d861726155de0ebd1a8e1e892f5af.jpg',
     'f375e6363bc21dcd3cb65637c7855e9c.jpg',
     'eda840df78ede0c000cd97b337b3d0db.jpg',
     'eed0a81e48da6883e1f3f248b3d398f2.jpg',
     'ea6f0fba4d83f7225c539d44cf28392d.jpg',
     'ed38d69eab73e9a66526ee16231f687a.jpg',
     'ea8bb75412610ee545d8c026ae789f4d.jpg',
     'e9efc5edbe14eb6375a64b37db8b36df.jpg',
     'f42938973bccdb8f6f556ea160a9357f.jpg',
     'f3d77f8157d981c44a1290b57cc03ac0.jpg',
     'f3929a0a7c3fede11ceb5d9e379778db.jpg',
     'ed119bc36eb02fbc0ed75e4d04440e0a.jpg',
     'eb5635af7d2f315b0f256aee4f41524e.jpg',
     'ed52e8685e7befe1a346651c0f9aaeaf.jpg',
     'f37af9dee180ed1195985decf4ef7111.jpg',
     'f52e47ff39ef157d700d51d62bea79d0.jpg',
     'ed1957de19321d3f09348c0c1e3321a8.jpg',
     'ee6479969035687eb6decab728718656.jpg',
     'ed650c72271b58dbac0fd5d91ebd206b.jpg',
     'ebe9a419fc7d2c67a934411960b88913.jpg',
     'ebd981f1b06aed2a15a1de27d9f2a5d2.jpg',
     'f56e24e008d532831fa5fa006618025b.jpg',
     'ed6e570e10f027ff20f0ab3598153193.jpg',
     'edccf5a23ce769b435817274226fe61f.jpg',
     'f234c1a8c84a833c77a7774da09cbb1a.jpg',
     'eadb3e22a37eb409670a69c6ad4de19a.jpg',
     'ef22e67b6d82c04f2d17249e42769d35.jpg',
     'f320c79c2f0df7a61cdfcf689f8dc91b.jpg',
     'eb1c2f4061d66878d2de2ce45a2382e5.jpg',
     'ed87717fd2068ed2c909afa65a2cfe2e.jpg',
     'f6d0770d25962d54d0cfedbfe14949a0.jpg',
     'ede38d1fbd47fcc1c408fbd7f0084fef.jpg',
     'edcfee2eaca6697d103ad03531aec0c3.jpg',
     'f3ffda74c6fcd4a8ff546bac62060276.jpg',
     'edd8796ad2ee790f6ab20f65329007c3.jpg',
     'f243cb6ec376a19f7de80e9c1a248518.jpg',
     'f2b70abdaeaae871fba924ab80d3253e.jpg',
     'f4f3f707907977e9f619540a51e6d9fa.jpg',
     'f6175a0ef289ef111b225a84240f2e4e.jpg',
     'ec14673c74fc8c62ee54ab67c9daf010.jpg',
     'ef0ca2e519acbcd5f7f03a97acdf0ef1.jpg',
     'efec77ba29d2407d82f808cf7ca79eb9.jpg',
     'ebeb2198b1392407fc54e13a9aa7cd0c.jpg',
     'ee10d8e27f4333c12c821c0584c02fdc.jpg',
     'f441dc37af37eb01c9fa3e0950228642.jpg',
     'ef14fa3af6b0cfb89b4e0ec98f75e586.jpg',
     'f6c0c3d4f34faf5caa5de6e81305de3b.jpg',
     'ea28a7ca3ddea1be196b48623e1ecd4e.jpg',
     'f588238227e5822b0c83a8355eccd067.jpg',
     'f19ca47ee303f9e770a298b26c993ec3.jpg',
     'ea668eb722f6167ae2669b6f1ddeb3d6.jpg',
     'f573d2958310513ce7113099598c7707.jpg',
     'ecf4a6af2c19e363bca3324a99cfecff.jpg',
     'eccb17b497007b28217bda18db73cce1.jpg',
     'ece034fecdb47d3e30115e7b70dd36d1.jpg',
     'f3e172b6c77d1d5b191722de535bc726.jpg',
     'f343a8b0d587a5f79c7da06732fcb347.jpg',
     'ece3ba1b205d68f8860612f993ade240.jpg',
     'ef71aaad573e1e289ec9ce92d844f96a.jpg',
     'eec0067218af5d0cca32a504501b98f1.jpg',
     'ef1bdfc864035ed213fd676c24ba14f0.jpg',
     'f0424f62ba3e580e9f34de645fb7cad4.jpg',
     'f28c40401e84e41f5aa4c60fea02a575.jpg',
     'eb53813edf96c446e5acaa06c0deca9e.jpg',
     'edce5253f5d09d42011e5b0f84d1831b.jpg',
     'f3b343a1d0a394e5d29a74f31f9922c8.jpg',
     'f1e4f7a6a490bd1bb2bfcb5586902e9e.jpg',
     'eeb4aa1fdbefcc783845be89b94bdab9.jpg',
     'eeaccf7f12c1e959de200bf2849ec101.jpg',
     'f64d8dfc5544c9fb4bdc578a7756dfd7.jpg',
     'eb2f7c626e517b63a1fc07577d8db675.jpg',
     'ee024b9ec41146caca0fd51b4acad8c0.jpg',
     'f5020914618ca6445b1bb1ca2e0a4d51.jpg',
     'f10aebe770d5533a5f7224a0bd816716.jpg',
     'f6430905d7328cb19856d0fe633f6493.jpg',
     'f547c95af25d09dc444943724396a68a.jpg',
     'f259fce0c617f40d26b75947f71f489c.jpg',
     'f3d517cae12f545fd445a0cfaf94b505.jpg',
     'ecccfab92b35ee51d1896eb35f5379a7.jpg',
     'efd68fb69d154ed0ff2dfea8f782f0d3.jpg',
     'ec01f50fdb5ca8749c5ee350232eceba.jpg',
     'ee307ebe9e98bac8b8a47be0d4803ba1.jpg',
     'f684cc75fb7996f315832975a3e07f67.jpg',
     'f056f4655ec9b7b942c1fcd6d4ea61f5.jpg',
     'ea1527f051be904e70fab5877bbc0341.jpg',
     'ec3fd4eea9a6a2c88908c33737442e4a.jpg',
     'f0efab499435ecd4e634b72f539fde9b.jpg',
     'f61517f58aa8a9d4248174bd75d891b1.jpg',
     'f2b5ba1f32494b942731fb5248e114a8.jpg',
     'f2cb6f6dfc542d47bfcea63b0dd78c6e.jpg',
     'f4a9095dabad1799e9747f623bbe0e86.jpg',
     'f2da0cd3a1f77cc96505d387d9dd731b.jpg',
     'ecffcf50e9df121652758b9371ca3792.jpg',
     'f3c2b2523885755ccfad67ff51308c98.jpg',
     'f32e24c17e03d5eb499875387f049ca9.jpg',
     'e9d31fe3a42451dc01c6d2e15ba6ef8d.jpg',
     'f3e83444819548799b1f77ac363a4d4a.jpg',
     'f52a0aaff41a517be216bf41c967a751.jpg',
     'eec01ae653b0373466a2719f8d56d8d1.jpg',
     'ec1b8f7b1a012af943ca2ec8efc58fb8.jpg',
     'ebff7c03c37ce6907387b62c68a1c138.jpg',
     'f1842cfc3bdbf485847574cb1b133d29.jpg',
     'ef029489fc2941792a5460afd7345fcc.jpg',
     'f23dffefcd4ae2dee8a5dbf67e41b699.jpg',
     'eeeb378e507ec043c3a35ddeb05c8536.jpg',
     'f430cdcc2b98a6e3566e71b27ae60e71.jpg',
     'f0634d128e68061c0220f224c9ba1228.jpg',
     'ed38d865e5ff611d6d1529652cc6464d.jpg',
     'eaea761df48d7adef74af9c66a0fbf4c.jpg',
     'f485d0ea64f8950b38954c17cfd03d8f.jpg',
     'eae07088a41c3d064b7d099b2fe63f11.jpg',
     'eb4ddd17cbdda67c5bab81f6407fc1ba.jpg',
     'ef29bc9846ee0f3599ee6c3791e5eb30.jpg',
     'f50c3a2915a6744f66237534db3d7779.jpg',
     'f01dce3264ed34d681815a9d0b03f7fc.jpg',
     'f58c4c4e6670e74fac63514cd9ac8ab9.jpg',
     'f5a88560838b1fa960f2bdf8bcc33621.jpg',
     'f0779cbe759a82f4927f60e47c9f1c81.jpg',
     'eccc1f78f99e1f5fc1fa618be4ae7f67.jpg',
     'ec3ca2186332da9d6bcd96b0caab2699.jpg',
     'ebb8c99c50ca5b48e010f0bda9a62c85.jpg',
     'f220628a83618e6cc13e78361e833361.jpg',
     'f5171a2251d123f01566ace3a1107754.jpg',
     'f39fbe4774a9b7662cedca6f9759e023.jpg',
     'ecda5afc51397fea7339c68c7bdaf3c9.jpg',
     'ebb2555fb23a8fe20c44050380154456.jpg',
     'f6b88d6bb18f14a24c8cd24b941cf0bf.jpg',
     'f3b47556eea0151354f1ee1b40762fce.jpg',
     'f184f4ea0aad4cac48670bede5868054.jpg',
     'f1f8d1fad725a613f6e841d3f49ee415.jpg',
     'f20c9c183552b734bc3c5904bfa32d09.jpg',
     'e9aaa8964f450f990a7a3c4228bb844d.jpg',
     'ee2567e0e8424a216c45b34682a80472.jpg',
     'ecd2e5133a9dc2877d9d7d77581edb87.jpg',
     'ef0cecbeaee4a08448d449a7c1e65f17.jpg',
     'ed142580e8aef16f8bae1e29381d807a.jpg',
     'ee41243497e7715fe3094324b5aceeda.jpg',
     'f2d229962b94228f494c9cb3c0ffe740.jpg',
     'e9da345a17d3e21041214e6cef9556a4.jpg',
     'f5feb8aae47a107bc2e8ca4075286ae5.jpg',
     'eaadc51c7c1a507fb380909113036428.jpg',
     'f0019323b5b8b321160c8199bea41118.jpg',
     'eaa2b8ce4fd5320fdeceecbe804f08bb.jpg',
     'f008c04f6d72140e1c40b8cf6bfa21d1.jpg',
     'ec8f1e4f39702089c985261d50c0c29f.jpg',
     'f1710a76fbc4ac9d47561e2b48dccd11.jpg',
     'ef8c805d9168bdcde66239d35834dd88.jpg',
     'eb909c348925d451cdcee84eeb21d15c.jpg',
     'ec53887c5887ec7be3693459ad3ba4ed.jpg',
     'f4c3afa01a61b3471656b42f8d5cb1e1.jpg',
     'ea906db2c833f5140b009eabba209eaa.jpg',
     'f6a32d6b0c663f2f949520149232a329.jpg',
     'ed61a3a5a304bd2f4e7038d8f95248ea.jpg',
     'f5d4b4e4770d3922b6b26dc46bdc57af.jpg',
     'ea607b36bd3391a5b4fe4c547f5dd7c7.jpg',
     'f6bdbf94ffea753f4f7638784faf01cc.jpg',
     'ea7f0c67bf41d5f2afe2e8e5a9c08774.jpg',
     'f4d535251be4d6195c3fc5d6a2c31e03.jpg',
     'f193aef29f983b44a4458c03ec309ca1.jpg',
     'ec760d9e97fdc7816c3cb44ea9aa33ed.jpg',
     'e9fc49ac28d4f84579f5621bccc78470.jpg',
     'ebe9487f88c13d27fec7db2592adf044.jpg',
     'f6e3a909254785d410b2418647034a5a.jpg',
     'ef16bf875e06977bfe0d4059326afffc.jpg',
     'ebb21cae1e6090f99253d83ea5fbbdc9.jpg',
     'edafef97d647ec7c4b4e00ed5fd3dde5.jpg',
     'ea15dfc5ed49598f8ea14a3be59164d7.jpg',
     'ee4f333b5dbfab72ccd4225f81920e8f.jpg',
     'f21952d038bd3c41c89def1a1d92385e.jpg',
     'f382107c98a4aa6d80a5b6a6df18e669.jpg',
     'f14f1200ba75d75293d9e1ca432f90bf.jpg',
     'eada9c7e7e1d9fde5df22b28e0ca9341.jpg',
     'edeb88b340e0b46a7c04161adbede2f0.jpg',
     'ea9e3ca803a028a17317345e4fa75135.jpg',
     'edd3d0f1175568806a1a052ba2051fd7.jpg',
     'f56a10d2912140974338c6330e97e466.jpg',
     'f589bf0f5d13407f1729e8ee342834f4.jpg',
     'f4a20b0c57e1fc7214548b4556a467f5.jpg',
     'f5373dbe567183432408067cb9dc0957.jpg',
     'ea43c69e5c1321a6d05d0caaceb69318.jpg',
     'f6e186742c1885a3c4a66396195d8a18.jpg',
     'ee08cef430080eb10b1388795f5576ad.jpg',
     'f111b639af2c7bd5e7e91e7595913894.jpg',
     'ecd514c6eefa5bb8e0b1f71724483e05.jpg',
     'f1512dff4effc6df01f50baa1135c139.jpg',
     'e99886bf590f8e0cb3394f91e8920ae2.jpg',
     'f45060c4cac8b690aaa61a510e1d30c5.jpg',
     'e9d910ff8abc407c1536201210b40888.jpg',
     'f3c0501b67da21916640efb617967dd4.jpg',
     'eb64056a3ab7ac1e47bfde7c2e8bd518.jpg',
     'f1ca2ebd997146e85932ce3cc6e8f064.jpg',
     'edf5a678ccc0861f55342166e3190f1e.jpg',
     'f69078e04e3d5e8d7689820de78f69ff.jpg',
     'ee1ce2280494dc07f516970abee34d55.jpg',
     'f641553e92efc6824ca4b9cef837ba46.jpg',
     'eb9dedfc29135debc82e82f159c23fa1.jpg',
     'f1a18e29959bc5af735ebb5045f91b25.jpg',
     'eabda7b78d4f4c71ad5686ca1589e8ad.jpg',
     'f3f295d33149977b1d67b8a3b6484ea7.jpg',
     'f25b4df00f4afe17ad37fb49a86c9ec1.jpg',
     'f0aa6b58cc701bbf31d090426cb4ecb4.jpg',
     'ea6c7f6e749d8aa9512054c47585e298.jpg',
     'e9d78bc07d863ba45ccee1a6c872182d.jpg',
     'eec7468f173bcd82691172de4f8233d8.jpg',
     'f055e34fcd3c7a941868a371821feb14.jpg',
     'f66de1bea3790719f70ef37779889a02.jpg',
     'ece94a0e987ed8316c4b3e4ed9e49477.jpg',
     'eb1f87344865dfe1e89717a7e82ae18b.jpg',
     'ec4ef1518f6b572f23b9bf298c0b6081.jpg',
     'f0b08d4f04e66b73526313b7cf0a6b08.jpg',
     'fbcc2af005aa6fa3c9cab4d1ac38bf06.jpg',
     'f842f546bc73b3c40698bbf9f94928dd.jpg',
     'f80541bbc3592999ccc617bdcbd9fde9.jpg',
     'f77d5b8a287c1395356e23849a2205fb.jpg',
     'fbd75ba5bf5ce7ef08a11a69c5ef263c.jpg',
     'fbcd86965ce247c0a44e50ff7b66a670.jpg',
     'f843b7f86256f4d603b1768c8b16adcc.jpg',
     'f7886847e4058293245db300e17928fc.jpg',
     'fe426e0af99930c0ec3c9ab58b02f8dc.jpg',
     'fb0c9c4e476ef5deeb432e2326c1f3da.jpg',
     'f90e5fa3f7dabd3292fecf4312ba98c4.jpg',
     'ffc1717fc5b5f7a6c76d0e4ea7c8f93a.jpg',
     'fdccec2dc716306a12b773e7689887c0.jpg',
     'fb2f2e9ccc7d47d475dcca8666d887f2.jpg',
     'fd5c9929ec93b09977a9565bd94b5672.jpg',
     'fda9aa966f3391e2508cdad4b51bbff9.jpg',
     'fd0f827a0d0ad9a2ff60ddf30e5d50be.jpg',
     'fa607eaf5aa5a95f58a6c43df1147e07.jpg',
     'febcab8eb2da444bf83336cffec7eb92.jpg',
     'fbaabd8210413b2084743c1f00d8c0f4.jpg',
     'ff6f47aa8e181b6efa4d0be7b09b5628.jpg',
     'fb7a49c3ed2c5f1a82d562fe0792c3fb.jpg',
     'feb16cf86c9dac6d476e3c372ba5c279.jpg',
     'fc6df38f35361d2630568b993a596008.jpg',
     'fd7eeee1c55efbb222223c2fc0b1bbbd.jpg',
     'fc0d67099704bcf210316e6c31d2f28b.jpg',
     'f97b9b8848683ab240cffeda2c855502.jpg',
     'fcab59cd421ec3d233da0027d664657e.jpg',
     'f835a7cc45610f664b91d252d413ff65.jpg',
     'f8e530f56e73403c8d69d29fbc391a06.jpg',
     'f9698ad44b009757cfacd9f171310651.jpg',
     'fefb453e43ec5e840c323538261493bd.jpg',
     'f7d18ca76e6b4e359a27955726e9a656.jpg',
     'f89b4055956d7574c6597a92e78ee149.jpg',
     'f75b13d14f950f2a6acf1f907bdfc636.jpg',
     'f89aacf1761449fc76d665cce1399f33.jpg',
     'fd73ff7c6b24e7292701305e8c7c32c2.jpg',
     'fc6e76acd21ed78f2c4c2153380cf63f.jpg',
     'fd03162f0a586bbeac13c33e74d24e73.jpg',
     'fd21ff6aa4f72aa6c29a589b2d3042fa.jpg',
     'ffd25009d635cfd16e793503ac5edef0.jpg',
     'f7c22e91e22c5a97abb2b7d9bab98809.jpg',
     'fc64ecf1ef29b56646e2480196877f7b.jpg',
     'fd05ad1ed6a60452b991ce65d735416a.jpg',
     'f9be292e5e04c9ff629ead6c17d5d87d.jpg',
     'facdc30ff148a0154121f1e17cac86d7.jpg',
     'fe7ea4eb63ab5fddea120555790f9187.jpg',
     'fc2c6cb0b33fec818063b2840217c546.jpg',
     'f9f2c52b196c9da8c7c98c68003d4f08.jpg',
     'f7933e6c90808d5bbbade7bde7cce6dd.jpg',
     'ffcb610e811817766085054616551f9c.jpg',
     'fa537c93c484404490f0bdd71c79ab6c.jpg',
     'ff0def9dafea6e633d0d7249554fcb2c.jpg',
     'fc4afae2a9dcf48ee66253add0709191.jpg',
     'fab313f7c055d30e79a8590ab05d54f3.jpg',
     'fcbad8ad476b83b180c9497bba2dedba.jpg',
     'fc440d7e1cd648dd5dac1ef18c05a799.jpg',
     'f84d455704632ab197b47cfe4061da36.jpg',
     'fd99c29a3ee91c7e9114e34be1ca0d1e.jpg',
     'ffbbf7536ba86dcef3f360bda41181b4.jpg',
     'fb1f17d411a56a2740310b4c06595877.jpg',
     'fb6a9896061f29a967d802e4f6dbbd74.jpg',
     'f7a44b90a798d72d636b7f1003c8c3e4.jpg',
     'fc71c9e6334f37a9f42c00e599af8b02.jpg',
     'fcee07f3f1190697b6ec35911dcdd8cb.jpg',
     'f75b457706d15d23fb803e2ad7c66c40.jpg',
     'fbd5762e97073435f8b7942fc67a4d16.jpg',
     'fa69bf620cee03a93e32188ff3872c43.jpg',
     'ffd3f636f7f379c51ba3648a9ff8254f.jpg',
     'f74471ba376e55adf638f527eb794c2a.jpg',
     'fe624532170510bd80627c0500bafc97.jpg',
     'feb9d0ae525ca28aabff74b455e34c16.jpg',
     'ffca1c97cea5fada05b8646998a5b788.jpg',
     'f776961d52ba3a7112506389623c8586.jpg',
     'fca387fcb6630ae11536b3374d3b2234.jpg',
     'fa7c0ebcade9a63facc59f32f96a5c17.jpg',
     'fce950ef7cf99efcb26297537bdf5dd4.jpg',
     'fdcf75632c624d8d03b37ddb1c6fc592.jpg',
     'fc3100b4cc97f41f55e86668d47d26e2.jpg',
     'fb23c7c26cc49324b9081e2086b32409.jpg',
     'ffa6a8d29ce57eb760d0f182abada4bf.jpg',
     'f9bf12ee2c9856d37d12b6299af87f8e.jpg',
     'f8a5dcec60ad103637c39b31d6dadaef.jpg',
     'fb542f35faf7843778cb2c35d81a7a44.jpg',
     'fbbae3a9f939903eb81b3d86ca4a8786.jpg',
     'ffa0055ec324829882186bae29491645.jpg',
     'ffe5f6d8e2bff356e9482a80a6e29aac.jpg',
     'f7627680c56c5d3acc4f7eae93124459.jpg',
     'fef5d4cdaf50cf159102e803c7d6aa9c.jpg',
     'f918d591dda3b7c1490f70dc92a7fb8e.jpg',
     'fe7171353417898022361453894adf94.jpg',
     'f988f65f2b26404fa16698ed835ac08a.jpg',
     'f831a464cc42602094575fbb18ff1143.jpg',
     'f95ac23ee909194006c538c4fc6e6750.jpg',
     'fdcedfa9f9ae621a4889e844b9e2940d.jpg',
     'f9b6c793a3720605ed68242e73d228d9.jpg',
     'fff43b07992508bc822f33d8ffd902ae.jpg',
     'ff2523c07da7a6cbeeb7c8f8dafed24f.jpg',
     'fef9c3ab585ad3f778c549fda42c1856.jpg',
     'f89759123d85d4135ed4bdce6923ad6a.jpg',
     'f819a72296da2b1dc4f21d8145c40f27.jpg',
     'fb6bca9a75ead518333fd3c98aa4dc40.jpg',
     'fad30cde8ff696cba83278a55a8de87b.jpg',
     'f7a8a885e2b28630634c3fb513277f27.jpg',
     'ff7d9c08091acc3b18b869951feeb013.jpg',
     'f7fdd6d141e45d148eaf6ba595a9c45b.jpg',
     'fb1d0dffd97612ee562cc17a98b683e4.jpg',
     'ffa16727a9ee462ee3f386be865b199e.jpg',
     'fee1696ae6725863f84b0da2c05ad892.jpg',
     'fb80f89c84c6c2649964b574918e02d6.jpg',
     'f9ac470f280837eda01169eafa5f0d23.jpg',
     'ff47baef46c5876eaf9a403cd6a54d72.jpg',
     'fb72489572110ed6d3e563c45f3ecb3e.jpg',
     'ff05f3976c17fef275cc0306965b3fe4.jpg',
     'f955bc6cf429a3957922f4155689c857.jpg',
     'fd2fbb4ff4bfe775d0267f4737c67521.jpg',
     'f8169d735ce64c6b8e4c421d61b2d3b5.jpg',
     'fa5df9801dec89450f56f373b074518c.jpg',
     'fc58ff2eb1f57943e151704a4f061d38.jpg',
     'fab4a52198780919e77a541aec737484.jpg',
     'f72df3c3daa677aa76027366ad55721f.jpg',
     'f901b7392fff2c1c3991bb8f7b31bfe3.jpg',
     'faa111bd776cdece79144aad36558eba.jpg',
     'f901c393eca491f1c03b88b31fa956fc.jpg',
     'f750daa88a8129fd8af9e6b686698147.jpg',
     'fa35207eea3c0d9939c1df7c86222853.jpg',
     'fa3d5cedd5c63e5cc0c04acadf95cb35.jpg',
     'f7de9a7b53f6b0b40a61ea56c55943d9.jpg',
     'fe50bac6c389d137ea01c9cfc7346ca8.jpg',
     'f9f253c1bf65d4ad1907faec9328eef2.jpg',
     'fcb911ca6411383f99bed4b745f4fede.jpg',
     'f8e53f2c3cdcdd5e42e7337d5b5156b4.jpg',
     'ff91c3c095a50d3d7f1ab52b60e93638.jpg',
     'f821311972e25d3dc8e8cf76a64e53e1.jpg',
     'fc11c7fdbef74c3a3d8ea731ec48861a.jpg',
     'fa289147f856e2525ee70529a6d0ac52.jpg',
     'f9d6e1b8bc0906d38bab497a228af52e.jpg',
     'fd8d47a35e1c84795b4a2a27b761eb7b.jpg',
     'fc2049d582b3444ed99af4a5c13b49e5.jpg',
     'f8c3ce8448eb8b1a4e5cb611595be906.jpg',
     'fc4f1d94ed5191b9b7d1f24420bbb07b.jpg',
     'f9dece751a4afe2330c02007681a7f81.jpg',
     'fdfcc3d2e40970fbfb8521bd29e9fb4d.jpg',
     'fd116b902bf7e614cdb22e725af87e58.jpg',
     'f7cd9846e7aa0c163ad98ad00ce51cf8.jpg',
     'fc9d449fb3c5be2e08839903d9405b42.jpg',
     'fbf881c1bd9b236af37efdfeb4a1fb51.jpg',
     'fc6abf69e1581b95734830af88c636a0.jpg',
     'f8d6ec246da32576653776342d52cb69.jpg',
     'fee672d906b502642597ccbc6acff0bb.jpg',
     'f9dd329f6a9df4dbad0d100aa2df0fd8.jpg',
     'f70f325bf6fc8aad05ed7a99212660fc.jpg',
     'f8df586a37584e859e252778a221391b.jpg',
     'fc11bd87c4d826650463f0ee79b8b4eb.jpg',
     'ffa0ad682c6670db3defce2575a2587f.jpg',
     'f738d8472928f06c5a3743914ac5d458.jpg',
     'ffc532991d3cd7880d27a449ed1c4770.jpg',
     'fa0561b43695b11a89447ae47afdbfeb.jpg',
     'fa0a112efe3604938af42b371c6227da.jpg',
     'fbdeba9ec017fc43c814e2688ad62402.jpg',
     'f8f529ee9da3f9edc72983ee020f3fbc.jpg',
     'fb9ee245256bc5c3e14f32c8c69cf6a8.jpg',
     'f7a9426876f70330f92f55cca08b19cd.jpg',
     'fb9c8b7e0b70b201898d930ed71163b6.jpg',
     'f74b4a01eae15a3d909c643e116bcfca.jpg',
     'ff52a3909f5801a71161cec95d213107.jpg',
     'f8c92d371c9f07b52bb6292fe4bdcc3f.jpg',
     'fe081bb43a6b0902d7ae9cb560053bc1.jpg',
     'fa6e79c819e06406740a88773b7f035c.jpg',
     'f84472e35734f1c9eb3be0372ba522aa.jpg',
     'ff4afeb51a1473f7ba18669a8ff48bc9.jpg',
     'fc0020cc00e3b1c7ec453ec129e17838.jpg',
     'ff181f0d69202b0650e6e5d76e9c13cc.jpg',
     'ff63fa05a58473138848f80840064d23.jpg',
     'fea60fdd28de5834520134d6dc77a9a2.jpg',
     'ff12508818823987d04e8fa4f5907efe.jpg',
     'fb9cbd60c09b65c1032fc7dab93c8354.jpg',
     'fde78cf59b95570111e6f851e06900cd.jpg',
     'f97da1d8c57873986183092ec4d5fc13.jpg',
     'f71e0bbcb7b9d348986393612e1b6800.jpg',
     'f811a137cac1489f074529064f79acfe.jpg',
     'ffcffab7e4beef9a9b8076ef2ca51909.jpg',
     'f78e0f62a1b5aad9da3a769c59121ed7.jpg',
     'f72555509a6ffd85394b1e6417c728df.jpg',
     'f8cc0c4e9be6827494a77260d383884f.jpg',
     'fa3bc3e096a2967f26113992b29b23b5.jpg',
     'f8009b210e1fca906bbfd55a17fcb224.jpg',
     'fa85c1ccb2fa7ff39d78bf2b821bcdd5.jpg',
     'fa3399d2f5241d807e0942b9807c2b4f.jpg',
     'fb119fab818c9c46445105a0e05a1fbf.jpg',
     'f7df3e61cec435ef6066e9d18b5c17b1.jpg',
     'fd00c84695d64324cc597937320cc0f2.jpg',
     'f888b4d0dac4f5b3faf7a05a6ad01cd4.jpg',
     'fd186806d3d7f123d9a568bcde794f6e.jpg',
     'fd3990bdd4b541a0efa8748ff8a901ac.jpg',
     'f881a7569fa39a96ce68e8c5c8642b20.jpg',
     'f8972355cef9e75502f9a79a26ba9798.jpg',
     'fa7d15c8d452895566586913f123a947.jpg',
     'ffa4e1bf959425bad9228b04af40ac76.jpg',
     'fa6247111ff3f3cc66713b4bf7c7d1b9.jpg',
     'f8e24c6d9d34d762c90c02efb45a96a3.jpg',
     'fa5054c5187c7171c546bd6a46b8346f.jpg',
     'fdcbedc65f600f81df181e9a46858cbb.jpg',
     'f706682a30021cc74cd9416dac25e943.jpg',
     'f9bf654be6ec7d1c40935aa397a3edbf.jpg',
     'fa4387315a7d0a81f033dc647546220a.jpg',
     'f7f73ba72679f35d552ef1af03c63bf5.jpg',
     'f8a46f8751e34aac4b262fed49f210c6.jpg',
     'f9c6acac2e566ae605ed9c7c1ecd6450.jpg',
     'fe4d298d682a42714f33085c9d241cc0.jpg',
     'fca2032135773108cb9811ab71d7f5c6.jpg',
     'fb4dbb9921a74bb2bab11c10a123971b.jpg',
     'f9900b44f49e5075c5b3fb589d1aae4b.jpg',
     'fab162b7c732afd2e0886e1fb158862f.jpg',
     'fcb2147b7999bfd687d4b87584bb8907.jpg',
     'ff0d0773ee3eeb6eb90a172d6afd1ea1.jpg',
     'fc5ea05e270b116e5c033f0b9e3b6082.jpg',
     'fc416bf2011ddccc885698a7ad19ba11.jpg',
     'fcec543b7b4dc1fc1e47d66a9305e87a.jpg',
     'f9b4e4ed6e77eca2ea940b31e58b408e.jpg',
     'fa96ff3a3b523de0b176e052843fd152.jpg',
     'fcccabf55377e660accd9c8bf984026c.jpg',
     'fa79fd049300f5082393fa99a142cd7f.jpg',
     'fdce4d488a629164ee8a9f2a0b81905f.jpg',
     'f9a2d5a36d32c2c1285cf46fe05972cb.jpg',
     'fa27c3d52a9c8a3c6f6801e92297963f.jpg',
     'ff84992beff3edd99b72718bec9448d2.jpg',
     'fe8d52ab96ff238ea7d234b508010ece.jpg',
     'fb56acfbe4b95a0df7a4b9e6bddbafd0.jpg',
     'fba676383a9e3dd93dc1c4d50c54f48f.jpg',
     'fbee719275e23f753f3cd1d9f83db21f.jpg',
     'fbc14efa3d420a5d5f4719a356446ad4.jpg',
     'f8e931fdc022ea586f2bcd50ca6de106.jpg',
     'fcecf02b9cc36ab7c4db440058f331c0.jpg',
     'f7070114b6f8cb965ba2dfb23b511b18.jpg',
     'f751bb00d954b8e4e132c958e7117011.jpg',
     'fef4a58219c8971820a85868a7b073f5.jpg',
     'fa8442a15275571b324a9e9ad6c7f3ed.jpg',
     'f83683931d1c95c7501b11a3edb4acbe.jpg',
     'fbb86c8f93d2a1068520c0363a2079bc.jpg',
     'f6f4398177ec0db196ab4ae92d79c6ea.jpg',
     'fae9b4b30924265a3098d8e524fc47d2.jpg',
     'f8d289525782f3caa8cb51627063ef22.jpg',
     'fb73518d7dc9e11b00c978d23115b8c6.jpg',
     'fba73e53151cb751f9c22e63de669923.jpg',
     'f921b60576b1055884307f1f6d42997f.jpg',
     'f8a16c174f672a88dff675bf2aa5e41f.jpg',
     'fa747f34cc4dc45dffd28135501c6688.jpg',
     'fb0568b6c25ef5bc9dded95154b1cfc9.jpg',
     'ff04baf19edbe449b39619d88da3633c.jpg',
     'fdd4ab27e77fe219d73a83dbb5123f97.jpg',
     'fce0a4c8bf2d6588ea7f6ecf731b2fbb.jpg',
     'fdc0556f9b32a714b56985eb99fbbbeb.jpg',
     'fab62c77c1af3606f359f987ed9b844e.jpg',
     'f9c845c69830b3682fbbbcc3fc139b44.jpg',
     'f78e04ee7d70acdb10b2079b25471a46.jpg',
     'f9561a534df6add1af1806bfa1558fde.jpg',
     'fa24ae09f6c25172009016b9dbec322d.jpg',
     'fc2d73fcdab269bde56ee0756e40d0c6.jpg',
     'fc0b8fd26e1db206b60074b7b2598465.jpg',
     'fd9aa3277a9635ecf6abbe637a78e521.jpg',
     'fbb02fe9f6a18d5e59c93e957dccf4fb.jpg',
     'f8271e8e638b9bab6a0d3e164874dd53.jpg',
     'fd082fe869dc90cf5fc21bfacc270265.jpg',
     'fb7160e2b5699b6c1548f2f7232f4774.jpg',
     'fd80cac3ffdcd6ddcfe4d2d6f8f617f2.jpg',
     'fdca814a411178296fe719d3c0be9049.jpg',
     'fb211e01f77eedac02e92d6bfeedb3ab.jpg',
     'ff3b935868afb51b2d0b75ddc989d058.jpg',
     'f7a5fbcc43b2e0e09fb25cb78a37094f.jpg',
     'f8b5c9e7e0aaa1412dcf01199e64c8d7.jpg',
     'f7be7c1d6af9e654f16e6d74ea63a8a5.jpg',
     'f81ec41ce2cc1acb16597fdd231c2406.jpg',
     'fa5d72cb50d3ab87bac4bedbfd45d3b0.jpg',
     'fdf83a2eed71f2c54c3e6f592e84b254.jpg',
     'fcb23e9c5e5915e021fd916e3df64ff6.jpg',
     'f8346f0bac3aae2f49113b3d7dabb812.jpg',
     'fbfb5da3096f8d12329dc9a02de83dd3.jpg',
     'f8ce0925091c9c81fec5ab16c1109f57.jpg',
     'f84bcbdda7152edced4b693e1ccaa888.jpg',
     'f824e27367e8475401b62b24111c467f.jpg',
     'f9eedeff3a54f28301468decc50e3def.jpg',
     'fc879f14ec130d86a2479e8d869908cd.jpg',
     'f99cdde623294ec669ff9f2a31e17830.jpg',
     'fd01857177284110f02b52475dfeb9cd.jpg',
     'fd229a951f5a9f2439d4f6c2c51595e3.jpg',
     'fe0ca31ba19fc97e9644ac4daafa7e36.jpg',
     'fcac8d16408b0967a416431030a3510c.jpg',
     'f76f9724c66f6b62edc1ac44fecaa27d.jpg',
     'ffcde16e7da0872c357fbc7e2168c05f.jpg',
     'fe76cbb5f172387f6a5b72739852d608.jpg',
     'ff0931b1c82289dc2cf02f0b4a165139.jpg',
     'f72944994eecb182f41456a820149950.jpg',
     'fb2d408bfd49f5a720c3192f7a1cc519.jpg',
     'fae8e42fae61856a5bd13bd4ac88b8dd.jpg',
     'fc992fdbd5af203a39e7c1868fc69090.jpg',
     'f88b18233f76dcfb8dc49ab1820fa0dd.jpg',
     'fee98c990f4d69c6a8467dd0f0668440.jpg',
     'f80f575bb32f2cc1958ef092681b9ea4.jpg',
     'fd411242df68b9da572f8d044afa3c5f.jpg',
     'fbd70c8820a6fbcd21242284baa8ba27.jpg',
     'f8accfdc2c24ee99bca4433e33e2a975.jpg',
     'fdb58e4dd078f4343ce953f1c0bf2a51.jpg',
     'fda872df827d4ad271ad139a891a665e.jpg',
     'f9d4235a0740a33550852fe5393aa929.jpg',
     'f9b7736812c24f6de73012511303fb28.jpg',
     'f9a2077bcf32a5d9d33176f52cbcc65c.jpg',
     'fd22da43e0b930b27b38eb97a55f65ba.jpg',
     'fb888e2a6e2acca312352030cc8b24de.jpg',
     'fcc9c872e76f097b8fd02b56760e7d48.jpg',
     'f7c998238377fda6bb1d8622c5d65c45.jpg',
     'fb32a67c8f89942906d08973dbad87b0.jpg',
     'f8ed43ad1a94147b0a53b9cb3fb2d4bc.jpg',
     'fd4816b0b1bf94c4868d16da9ee87e10.jpg',
     'fd12fd971ffa3d489990b6eeca1952b8.jpg',
     'fc7cefece70681db1b55887a4b02a901.jpg',
     'fa1a19bd9f99b862cb0a986cbc0ea803.jpg',
     'fdb60e77166ba6aa999e5dab05c44dbc.jpg',
     'f8a9f218dbc135010a7099efbb36a8ce.jpg',
     'fbbe0a41f7fa5cd4b69f25d8693e55f9.jpg',
     'f708b2f18826cc754451e07de4ba148c.jpg',
     'ff0c4e0e856f1eddcc61facca64440c9.jpg',
     'fe54e87e65fe0c68670c0dd1a923f1f0.jpg',
     'fd42b1ea571fbdd77c1c5ceeff8ebb76.jpg',
     'fb5898e240410c7d736548bf938bbc0a.jpg',
     'f99c0ede8bf16d5e94a3a69ffb97b341.jpg',
     'fe9e09be6594f626f0d711bfba10cfe0.jpg',
     'fbf6ac0549525e721d3d3e48e27db4c5.jpg',
     'fa26802c7a0ff2fc7dcabb9999a6c4b3.jpg',
     'ff63ed894f068da8e2bbdfda50a9a9f8.jpg',
     'f811a192eb721accaa495e6722d9acff.jpg',
     'feafd0730eae85e63a41bbc030755c59.jpg',
     'fa2a33c1dc8b39ad51738408b289a0de.jpg',
     'ffc2b6b9133a6413c4a013cff29f9ed2.jpg',
     'fddb09d408084b1289bf572bd4071c9c.jpg',
     'fab341dadc50afc1a507a9910d5101b3.jpg',
     'fe49341352549164ad921a67647507f1.jpg',
     'f922fcb519e1f55ca99e8c3bb9e07619.jpg',
     'ff54d45962b3123bb67052e8e29a60e7.jpg',
     'fb9ad84feb0f23f26247056b830f6e36.jpg',
     'f8d48f89aaa55962d4beb853a128eac7.jpg',
     'f827aadcd7537def18615dc19f650456.jpg',
     'fc518958bdeda396c5431d8a8ec319df.jpg',
     'f8387071cbb4e77a274ab6035c3bb687.jpg',
     'fe3d08ee9e1aba1785391b42345c3fc0.jpg',
     'fe0beca881efd723510e8e859306a3a6.jpg',
     'fc94a4822e2d9427568b6d5427e903a7.jpg',
     'f8068ff990794cc54cd2d647387ecf6d.jpg',
     'fdc614c16f54555064a32bc94522b4a4.jpg',
     'f9b333457342a87a8321d8686035a7cf.jpg',
     'fa5296383aa39dd516c2f5610c6e71b0.jpg',
     'f7914989bd1e633c30a76054df77266d.jpg',
     'fcaff3d64f414b0a095d9a50cc29c401.jpg',
     'f6f603600c231d6dd529c54e45a7b2c5.jpg',
     'fa4fdbc06a4bf03494884f3dcb062db0.jpg',
     'fc7317da160bff89cd13aacc980adf26.jpg',
     'fa1e9d2b34ac79ba9b2358b2c0d803f5.jpg',
     'f94a145a3a291fc2c535d135938aea72.jpg',
     'f78c6a61f77eae6388af7f56c8b4aea7.jpg',
     'f988271af1ca0e21b76b8cb318bcf4b3.jpg',
     'f9024336c267f4dd4de82cf5f617ffa0.jpg',
     'fd64b07c6c3249ae625564fc111adff6.jpg',
     'fa6e79b065e3f95cf98430ed54700301.jpg',
     'fa1ae7cd7d26a1dbc29149daa8b03ea6.jpg',
     'fede60fb2acc02a2da0d0a05f760b7d5.jpg',
     'f89bc5490cbc1b17dc0be9c5bdd4f224.jpg',
     'fc77bf555c892344771a2c6714e72659.jpg',
     'fbf3162c4df3f1a527cb0b26ef062704.jpg',
     'ffe2ca6c940cddfee68fa3cc6c63213f.jpg',
     'fc4fc2e64dc59dcff402fc281a42481d.jpg',
     'fe03d2d88e9a68aeeac63f33a920557d.jpg',
     'f8c2b71ba0ee8fb0b64a589f12c98618.jpg',
     'ff8e3fa7e04faca99af85195507ee54d.jpg',
     'f7850cfce0e0d79627dfb63d71d97b72.jpg',
     'fccd6fe10febbc04d13470ec2aa516af.jpg',
     'fa72d780b780d1a8fda59852ab62738f.jpg',
     'fdca49e41439a428f8f727115e2610a9.jpg',
     'f7f60f8bad2179cf4b55b8800497eed8.jpg',
     'ff4bb57ce419cd637dd511a1b5474bff.jpg',
     'fe5e4ee18529af1af1861efd550561a3.jpg',
     'f9a664aff90ef04521974670cbd19cfe.jpg',
     'fab782d25875a7cf5298cd2e2aa01cd5.jpg',
     'fd3c4ef41e17c992e0c5aaa5101a39f3.jpg',
     'fb1ffb75a67173aa922f40c11c40264a.jpg',
     'fa4b9e80e2300c45a73f165fd34b3378.jpg',
     'fc338aa6f2ba965bb7be53584ca0b4e0.jpg',
     'faa2c24c801b37aca93ac744da51c2c3.jpg',
     'f80ad50586c794ea8555e3bed23e0c0e.jpg',
     'f98392bfae7d0aa35ecc4993ee2afbaf.jpg',
     'f82a2b46267b6d4edc62c76ebdb0fc8b.jpg',
     'f8a95fd0eee5042ed2b93e6abd032baa.jpg',
     'fdd70b075a651fb1a966e5a8735b9a34.jpg',
     'faf615fd8d21d9906b8ab5091e4f5b82.jpg',
     'fa0b4de9dbacd9faa3a5a4f2adc63195.jpg',
     'fb1cdd8ff249b6ee7c2af2e89ff644d4.jpg',
     'fe78fc42e32174c7178b572bdcf5a129.jpg',
     'f7564622d2ebbcebdb2eb1f150dbaa95.jpg',
     'fabd0085e9b6b9c83b9fb419e0a2b5b7.jpg',
     'f9a1339b0ef12e47ae875a86cf91524b.jpg',
     'fae43d1ff21d5a8847887f7c7695afa5.jpg',
     'ff7334b06cee8667a7f30eb00e0b93cf.jpg',
     'fd8bd9ce34326d7fea7b85040510bdbb.jpg',
     'fe13d46f5920f0944e6c30e54ac0e2a5.jpg',
     'fb8c009186d200c2f49173d89f4a4a80.jpg',
     'fe3e760d763e186541e18f303cd7caca.jpg',
     'fd2478ca48b9ac774babc2cae65fca5c.jpg',
     'f817cad476ff2a52fbe5b56a422dd577.jpg',
     'fc33f90570fc8502e6c3f83a6bf3b982.jpg',
     'f7266a5b52cf22b11274b5fe66a52979.jpg',
     ...]




```python
# one more check
Image(filenames[9000])
```




    
![jpeg](output_17_0.jpg)
    




```python
labels_csv['breed'][9000]
```




    'tibetan_mastiff'



Prepare labels


```python
import numpy as np
labels=labels_csv['breed'].to_numpy()
#labels=np.array(labels)
labels
```




    array(['boston_bull', 'dingo', 'pekinese', ..., 'airedale',
           'miniature_pinscher', 'chesapeake_bay_retriever'], dtype=object)




```python
# check if number of labels into number of filenames
len(labels),len(filenames)
```




    (10222, 10222)




```python
# find unique labels
unique_breeds=np.unique(labels)
len(unique_breeds)
```




    120




```python
unique_breeds[:20]
```




    array(['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
           'american_staffordshire_terrier', 'appenzeller',
           'australian_terrier', 'basenji', 'basset', 'beagle',
           'bedlington_terrier', 'bernese_mountain_dog',
           'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
           'bluetick', 'border_collie', 'border_terrier', 'borzoi',
           'boston_bull'], dtype=object)




```python
# turn a single label into an array of booleans
print(labels[0])
labels[0]==unique_breeds
```

    boston_bull





    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False,  True, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False])




```python
# turn every label into a boolean array
boolean_labels=[label == unique_breeds for label in labels]
boolean_labels[:2]
```




    [array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False,  True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False]),
     array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False,  True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False])]




```python
len(boolean_labels)
```




    10222




```python
np.shape(boolean_labels)
```




    (10222, 120)




```python
# turn boolean array into integers
print(labels[0])
print(np.where(unique_breeds==labels[0]))   # return index
print(boolean_labels[0].argmax()) # index where label occurs in boolean array
print(boolean_labels[0].astype(int)) # there will be a 1 where the sample label occurs
```

    boston_bull
    (array([19]),)
    19
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0]



```python
filenames[:10]
```




    ['/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/000bec180eb18c7604dcecc8fe0dba07.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001cdf01b096e06d78e9e5112d419397.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00214f311d5d2247d5dfe4fe24b2303d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0021f9ceb3235effd7fcde7f7538ed62.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002211c81b498ef88e1b40b9abf84e1d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00290d3e1fdd27226ba27a8ce248ce85.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002a283a315af96eaea0e28e7163b21b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/003df8b8a8b05244b1d920bb6cf451f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0042188c895a2f14ef64a918ed9c7b64.jpg']




```python
boolean_labels[:2]
```




    [array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False,  True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False]),
     array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False,  True, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False])]



### Create validation set


```python
# Split X & y
X=filenames
y=boolean_labels
len(X),len(y)
```




    (10222, 10222)



Start off experimenting with ~1000 images and increase as needed


```python
# set number of images to use for experimenting
NUM_IMAGES=1000  #@param {type:'slider',min:1000,max:10000,step:1000}
```


```python
# split data into train and validation sets
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X[:NUM_IMAGES],
                                             y[:NUM_IMAGES],
                                             test_size=0.2,
                                             random_state=42)

len(X_train),len(y_train),len(X_val),len(y_val)
```




    (800, 800, 200, 200)




```python
X_train[:2],y_train[:2]
```




    (['/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00bee065dcec471f26394855c5c2f3de.jpg',
      '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d2f9e12a2611d911d91a339074c8154.jpg'],
     [array([False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False,  True,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False]),
      array([False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False,  True, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False,
             False, False, False])])



## Preprocessing images (turning images into Tensors)

To preprocess images into Tensors, write a function which performs a few things:

1. Take an image filepath as input
2. Use TensorFlow to read the file and save it to a variable, `image`
3. Turn `image` (.jpg) into Tensors
4. Normalize image (convert color channel values from 0-255 to 0-1)
5. Resize the `image` to be a shape of (224,224)
6. Return the modified image


```python
# convert image to numpy array
from matplotlib.pyplot import imread
image=imread(filenames[42])
image.shape
```




    (257, 350, 3)




```python
image.max(),image.min()
```




    (255, 0)




```python
image
```




    array([[[ 89, 137,  87],
            [ 76, 124,  74],
            [ 63, 111,  59],
            ...,
            [ 76, 134,  86],
            [ 76, 134,  86],
            [ 76, 134,  86]],
    
           [[ 72, 119,  73],
            [ 67, 114,  68],
            [ 63, 111,  63],
            ...,
            [ 75, 131,  84],
            [ 74, 132,  84],
            [ 74, 131,  86]],
    
           [[ 56, 104,  66],
            [ 58, 106,  66],
            [ 64, 112,  72],
            ...,
            [ 71, 127,  82],
            [ 73, 129,  84],
            [ 73, 130,  85]],
    
           ...,
    
           [[  2,  11,  10],
            [  5,  14,  13],
            [  6,  15,  14],
            ...,
            [120, 113,  67],
            [126, 118,  72],
            [122, 114,  67]],
    
           [[  0,   4,   6],
            [  0,   9,   8],
            [  1,  10,   9],
            ...,
            [105,  98,  52],
            [111, 104,  58],
            [111, 103,  56]],
    
           [[ 18,  16,  37],
            [ 18,  18,  28],
            [ 17,  20,  11],
            ...,
            [101,  92,  53],
            [ 97,  88,  49],
            [120, 111,  72]]], dtype=uint8)




```python
tf.constant(image)
```




    <tf.Tensor: shape=(257, 350, 3), dtype=uint8, numpy=
    array([[[ 89, 137,  87],
            [ 76, 124,  74],
            [ 63, 111,  59],
            ...,
            [ 76, 134,  86],
            [ 76, 134,  86],
            [ 76, 134,  86]],
    
           [[ 72, 119,  73],
            [ 67, 114,  68],
            [ 63, 111,  63],
            ...,
            [ 75, 131,  84],
            [ 74, 132,  84],
            [ 74, 131,  86]],
    
           [[ 56, 104,  66],
            [ 58, 106,  66],
            [ 64, 112,  72],
            ...,
            [ 71, 127,  82],
            [ 73, 129,  84],
            [ 73, 130,  85]],
    
           ...,
    
           [[  2,  11,  10],
            [  5,  14,  13],
            [  6,  15,  14],
            ...,
            [120, 113,  67],
            [126, 118,  72],
            [122, 114,  67]],
    
           [[  0,   4,   6],
            [  0,   9,   8],
            [  1,  10,   9],
            ...,
            [105,  98,  52],
            [111, 104,  58],
            [111, 103,  56]],
    
           [[ 18,  16,  37],
            [ 18,  18,  28],
            [ 17,  20,  11],
            ...,
            [101,  92,  53],
            [ 97,  88,  49],
            [120, 111,  72]]], dtype=uint8)>




```python
IMG_SIZE=224

def process_image(image_path,img_size=IMG_SIZE):
  '''
  Take an image file path and turns the image into a Tensor.
  '''
  # read in an image file
  image=tf.io.read_file(image_path)

  # turn the jpeg image into numerical Tensor with 3 color channels (Red, Green, Blue)
  image=tf.image.decode_jpeg(image,channels=3)

  # convert the color channel values from 0-255 to 0-1 values
  image=tf.image.convert_image_dtype(image,tf.float32)    # normalization

  # resize the image (224,224)
  image=tf.image.resize(image,size=[img_size,img_size])

  return image
```

## Turning data into batches

If process 10,000+ images in one go, they all might not fit into memory.
 
So we do about 32 (batch size) images at a time (adjust batch size if need be).

`(image,label)`


```python
# create a function to return a tuple (image,label)

def get_image_label(image_path,label):
  image=process_image(image_path)     # process the image first
  return image,label
```


```python
# demo
(process_image(X[42]),tf.constant(y[42]))
```




    (<tf.Tensor: shape=(224, 224, 3), dtype=float32, numpy=
     array([[[0.3264178 , 0.5222886 , 0.3232816 ],
             [0.2537167 , 0.44366494, 0.24117757],
             [0.25699762, 0.4467087 , 0.23893751],
             ...,
             [0.29325107, 0.5189916 , 0.3215547 ],
             [0.29721776, 0.52466875, 0.33030328],
             [0.2948505 , 0.5223015 , 0.33406618]],
     
            [[0.25903144, 0.4537807 , 0.27294815],
             [0.24375686, 0.4407019 , 0.2554778 ],
             [0.2838985 , 0.47213382, 0.28298813],
             ...,
             [0.2785345 , 0.5027992 , 0.31004712],
             [0.28428748, 0.5108719 , 0.32523635],
             [0.28821915, 0.5148036 , 0.32916805]],
     
            [[0.20941195, 0.40692952, 0.25792548],
             [0.24045378, 0.43900946, 0.2868911 ],
             [0.29001117, 0.47937486, 0.32247734],
             ...,
             [0.26074055, 0.48414773, 0.30125174],
             [0.27101526, 0.49454468, 0.32096273],
             [0.27939945, 0.5029289 , 0.32934693]],
     
            ...,
     
            [[0.00634795, 0.03442048, 0.0258106 ],
             [0.01408936, 0.04459917, 0.0301715 ],
             [0.01385712, 0.04856448, 0.02839671],
             ...,
             [0.4220516 , 0.39761978, 0.21622123],
             [0.47932503, 0.45370543, 0.2696505 ],
             [0.48181024, 0.45828083, 0.27004552]],
     
            [[0.00222061, 0.02262166, 0.03176915],
             [0.01008397, 0.03669046, 0.02473482],
             [0.00608852, 0.03890046, 0.01207283],
             ...,
             [0.36070833, 0.33803678, 0.16216145],
             [0.42499566, 0.3976801 , 0.21701711],
             [0.4405433 , 0.4139589 , 0.23183356]],
     
            [[0.05608025, 0.06760229, 0.10401428],
             [0.05441074, 0.07435255, 0.05428263],
             [0.04734282, 0.07581793, 0.02060942],
             ...,
             [0.3397559 , 0.31265694, 0.14725602],
             [0.387725  , 0.360274  , 0.18714729],
             [0.43941984, 0.41196886, 0.23884216]]], dtype=float32)>,
     <tf.Tensor: shape=(120,), dtype=bool, numpy=
     array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
             True, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False])>)



Now we get the tuple version of data, let's make a function to turn all of data `(X & y)` into batches.


```python
# define the batch size 32
BATCH_SIZE=32

def create_data_batches(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
  """
  Create batches of data out of image (X) and label (y) pairs.
  shuffles the data if it's a training data but doesn't shuffle if it's a validation data.
  Also accepts test data as input (no labels).
  """
  
  # if the data is a test data, we don't have labels
  if test_data:
    print("Creating test data batches...")
    data=tf.data.Dataset.from_tensor_slices(tf.constant(X)) # only filepaths (no labels)
    data_batch=data.map(process_image).batch(batch_size)
    return data_batch

  # if the data is a valid dataset, we don't need to shuffle it.
  elif valid_data:
    print("Creating validation data batches...")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(X),  # filepaths
                                            tf.constant(y)))  # labels
    data_batch=data.map(get_image_label).batch(batch_size)
    return data_batch
  else:
    print("Creating training data batches...")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                             tf.constant(y)))
    
    # shuffle pathnames and labels before mapping image processor function is faster than shuffling images.
    data=data.shuffle(buffer_size=len(X))
    
    data_batch=data.map(get_image_label).batch(batch_size)
    return data_batch
```


```python
# creating training and validation data batches
train_data=create_data_batches(X_train,y_train)
val_data=create_data_batches(X_val,y_val,valid_data=True)
```

    Creating training data batches...
    Creating validation data batches...



```python
# check out the different attributes 
train_data.element_spec,val_data.element_spec
```




    ((TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
      TensorSpec(shape=(None, 120), dtype=tf.bool, name=None)),
     (TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
      TensorSpec(shape=(None, 120), dtype=tf.bool, name=None)))



## Visualizing data batches



```python
import matplotlib.pyplot as plt

# create a function for viewing images in a data batch
def show_25_images(images,labels):
  """
  Display a plot of 25 images and their labels from a data batch.
  """
  plt.figure(figsize=(10,10))
  for i in range(25):
    # create subplots 5*5
    ax=plt.subplot(5,5,i+1)
    plt.imshow(images[i])
    plt.title(unique_breeds[labels[i].argmax()])
    plt.axis("off")

```


```python
train_data
```




    <BatchDataset shapes: ((None, 224, 224, 3), (None, 120)), types: (tf.float32, tf.bool)>




```python
train_images,train_labels=next(train_data.as_numpy_iterator())
train_images,train_labels
```




    (array([[[[0.35052523, 0.28778014, 0.15444678],
              [0.34961486, 0.27510506, 0.14569329],
              [0.34541318, 0.2741947 , 0.13333334],
              ...,
              [0.45882356, 0.36862746, 0.20392159],
              [0.46963415, 0.3715949 , 0.21473216],
              [0.46963415, 0.3715949 , 0.21473216]],
     
             [[0.34766284, 0.28491774, 0.15158439],
              [0.34675246, 0.27224267, 0.1428309 ],
              [0.3456432 , 0.2744247 , 0.13356335],
              ...,
              [0.46168596, 0.37148985, 0.20678398],
              [0.4666667 , 0.36862746, 0.21176472],
              [0.4666667 , 0.36862746, 0.21176472]],
     
             [[0.341833  , 0.2790879 , 0.14575456],
              [0.34424895, 0.26973915, 0.1403274 ],
              [0.3457283 , 0.27450982, 0.13364847],
              ...,
              [0.4675158 , 0.37731972, 0.2126138 ],
              [0.4675158 , 0.36947656, 0.2126138 ],
              [0.4675158 , 0.36947656, 0.2126138 ]],
     
             ...,
     
             [[0.7658783 , 0.7580352 , 0.8090156 ],
              [0.7612366 , 0.7533934 , 0.80437386],
              [0.756269  , 0.7484258 , 0.79940623],
              ...,
              [0.97604513, 0.43172777, 0.52129364],
              [0.97501194, 0.3973508 , 0.50775   ],
              [0.92699   , 0.34267625, 0.45640177]],
     
             [[0.71872795, 0.7108848 , 0.76186526],
              [0.7235203 , 0.71567714, 0.7666576 ],
              [0.7302407 , 0.72239757, 0.773378  ],
              ...,
              [0.95350885, 0.40824056, 0.48996326],
              [0.9671215 , 0.4006298 , 0.49926427],
              [0.9330354 , 0.35656476, 0.45852557]],
     
             [[0.6915527 , 0.68370956, 0.73468995],
              [0.7032737 , 0.6954306 , 0.74641097],
              [0.7191547 , 0.7113116 , 0.762292  ],
              ...,
              [0.9285909 , 0.37740445, 0.4531922 ],
              [0.95909023, 0.39394984, 0.4921339 ],
              [0.93621325, 0.3686452 , 0.4676385 ]]],
     
     
            [[[0.39326304, 0.36144006, 0.20753121],
              [0.2790592 , 0.27029297, 0.07393364],
              [0.3111188 , 0.3380929 , 0.13875364],
              ...,
              [0.6524881 , 0.71131164, 0.8013544 ],
              [0.6387653 , 0.70151037, 0.79954964],
              [0.6304547 , 0.6931998 , 0.7912391 ]],
     
             [[0.3542631 , 0.33214223, 0.18944265],
              [0.31817868, 0.321957  , 0.13625146],
              [0.2752798 , 0.29986838, 0.11687157],
              ...,
              [0.65830946, 0.717133  , 0.80097413],
              [0.6416436 , 0.7043887 , 0.7975722 ],
              [0.63338184, 0.69612694, 0.7941662 ]],
     
             [[0.3954986 , 0.39300144, 0.27144155],
              [0.29967126, 0.31351689, 0.15295625],
              [0.34518915, 0.373421  , 0.21534514],
              ...,
              [0.6682142 , 0.72813374, 0.79928744],
              [0.6433911 , 0.70685637, 0.7958247 ],
              [0.63463753, 0.6973826 , 0.79372376]],
     
             ...,
     
             [[0.7009822 , 0.75588423, 0.7935376 ],
              [0.7002754 , 0.7512558 , 0.78582984],
              [0.74666905, 0.79670405, 0.82839173],
              ...,
              [1.        , 1.        , 0.9921569 ],
              [0.9777574 , 0.98167896, 0.9620711 ],
              [0.95261526, 0.9565368 , 0.936929  ]],
     
             [[0.7502116 , 0.8051136 , 0.85256726],
              [0.75533444, 0.81023645, 0.8536953 ],
              [0.68513376, 0.73963547, 0.7771333 ],
              ...,
              [1.        , 1.        , 1.        ],
              [0.9844857 , 0.98840725, 0.96820414],
              [0.95816797, 0.96208954, 0.9385601 ]],
     
             [[0.7630439 , 0.82388085, 0.87713987],
              [0.7927362 , 0.85357314, 0.902935  ],
              [0.7540494 , 0.8083212 , 0.85177356],
              ...,
              [1.        , 1.        , 1.        ],
              [0.9931938 , 0.9971153 , 0.97691226],
              [0.9658015 , 0.96972305, 0.94619364]]],
     
     
            [[[0.13773258, 0.22346345, 0.11700305],
              [0.23252553, 0.30763057, 0.19563922],
              [0.18917102, 0.23913285, 0.12143797],
              ...,
              [0.62581116, 0.6604751 , 0.65875924],
              [0.51773745, 0.5334237 , 0.5451884 ],
              [0.52265537, 0.53834164, 0.55010635]],
     
             [[0.25052395, 0.3207971 , 0.23725367],
              [0.05130488, 0.10808291, 0.02970343],
              [0.12517099, 0.1568843 , 0.06729877],
              ...,
              [0.54057753, 0.5637918 , 0.5656825 ],
              [0.5297618 , 0.54544806, 0.55721277],
              [0.4893205 , 0.5050068 , 0.5167715 ]],
     
             [[0.37000483, 0.41180843, 0.35321406],
              [0.24726026, 0.27108258, 0.21825583],
              [0.2645443 , 0.27110162, 0.22206265],
              ...,
              [0.5270998 , 0.54968387, 0.55188966],
              [0.50706005, 0.5227463 , 0.534511  ],
              [0.51150376, 0.52719   , 0.53895473]],
     
             ...,
     
             [[0.33149698, 0.3085978 , 0.31612584],
              [0.35751867, 0.3346195 , 0.3421475 ],
              [0.37277928, 0.34988007, 0.3574081 ],
              ...,
              [0.48090404, 0.42992365, 0.39794618],
              [0.50177634, 0.44523203, 0.40264222],
              [0.49691883, 0.4311975 , 0.3916667 ]],
     
             [[0.2593196 , 0.23971173, 0.25539804],
              [0.3483598 , 0.32875195, 0.34443823],
              [0.3218138 , 0.30220595, 0.31789222],
              ...,
              [0.5139126 , 0.48765218, 0.4158734 ],
              [0.49932015, 0.45377606, 0.39955974],
              [0.51006705, 0.45890987, 0.41390026]],
     
             [[0.31691387, 0.29730603, 0.3129923 ],
              [0.31130943, 0.29170159, 0.30738786],
              [0.3424053 , 0.32279745, 0.33848372],
              ...,
              [0.4658096 , 0.44620177, 0.36777037],
              [0.47771463, 0.4406347 , 0.38359696],
              [0.49383715, 0.44677833, 0.3997195 ]]],
     
     
            ...,
     
     
            [[[0.09851191, 0.07890406, 0.05537465],
              [0.09042367, 0.07081583, 0.04728642],
              [0.07683824, 0.0572304 , 0.03370098],
              ...,
              [0.00740576, 0.0290446 , 0.05257401],
              [0.        , 0.        , 0.0145661 ],
              [0.00280115, 0.        , 0.        ]],
     
             [[0.09411766, 0.07450981, 0.0509804 ],
              [0.09019608, 0.07058824, 0.04705883],
              [0.09464286, 0.07503502, 0.05150561],
              ...,
              [0.01365602, 0.03340415, 0.05693357],
              [0.        , 0.        , 0.0145661 ],
              [0.00280115, 0.        , 0.        ]],
     
             [[0.09900211, 0.07939426, 0.05586485],
              [0.10509454, 0.0854867 , 0.06195728],
              [0.10190827, 0.08230042, 0.05877101],
              ...,
              [0.04427623, 0.07172722, 0.09525663],
              [0.        , 0.        , 0.0145661 ],
              [0.00280115, 0.        , 0.        ]],
     
             ...,
     
             [[0.20379904, 0.20772061, 0.18419118],
              [0.2079482 , 0.21186976, 0.18834035],
              [0.22165617, 0.22557774, 0.20204833],
              ...,
              [0.20647754, 0.2103991 , 0.1868697 ],
              [0.20495437, 0.20887594, 0.18534651],
              [0.21300769, 0.21692926, 0.19339985]],
     
             [[0.20514707, 0.20906864, 0.18553922],
              [0.21064427, 0.21456584, 0.19103643],
              [0.22872901, 0.23265058, 0.20912117],
              ...,
              [0.23891793, 0.2428395 , 0.21931009],
              [0.23774514, 0.24166672, 0.2181373 ],
              [0.22097337, 0.22489494, 0.20136553]],
     
             [[0.20560226, 0.20952383, 0.18599442],
              [0.21848741, 0.22240898, 0.19887957],
              [0.2296919 , 0.23361346, 0.21008405],
              ...,
              [0.23529413, 0.23137257, 0.21176472],
              [0.23137257, 0.227451  , 0.20784315],
              [0.22464985, 0.22072828, 0.20112044]]],
     
     
            [[[0.5197562 , 0.43592465, 0.38020837],
              [0.52778155, 0.4415879 , 0.386659  ],
              [0.51467323, 0.4306799 , 0.37770486],
              ...,
              [0.03707098, 0.03707098, 0.03707098],
              [0.03438376, 0.03438376, 0.03438376],
              [0.03046219, 0.03046219, 0.03046219]],
     
             [[0.52245206, 0.4561173 , 0.39376682],
              [0.5181481 , 0.44704556, 0.39210647],
              [0.50980395, 0.44049373, 0.39032742],
              ...,
              [0.03077731, 0.03077731, 0.03077731],
              [0.02689943, 0.02689943, 0.02689943],
              [0.02465051, 0.02465051, 0.02465051]],
     
             [[0.5121913 , 0.46089578, 0.39909607],
              [0.5016492 , 0.45035368, 0.38855398],
              [0.50464815, 0.45131314, 0.3971435 ],
              ...,
              [0.05119634, 0.05119634, 0.05119634],
              [0.04741413, 0.04741413, 0.04741413],
              [0.04466788, 0.04466788, 0.04466788]],
     
             ...,
     
             [[0.41655818, 0.3224405 , 0.3224405 ],
              [0.39437294, 0.30025527, 0.30025527],
              [0.39683148, 0.30271384, 0.30271384],
              ...,
              [0.3659803 , 0.26794106, 0.27970576],
              [0.39204592, 0.29400668, 0.30577138],
              [0.3854389 , 0.28739965, 0.29916435]],
     
             [[0.3956583 , 0.2807423 , 0.29982424],
              [0.36617652, 0.2512605 , 0.26754203],
              [0.38151613, 0.26741594, 0.28328958],
              ...,
              [0.35330763, 0.26703313, 0.27487627],
              [0.34582484, 0.25955033, 0.26739347],
              [0.3442134 , 0.2579389 , 0.26578203]],
     
             [[0.3989846 , 0.277416  , 0.3009454 ],
              [0.3695028 , 0.24793419, 0.26754203],
              [0.38256302, 0.26636904, 0.28328958],
              ...,
              [0.33872443, 0.25244993, 0.26029307],
              [0.32050043, 0.23422591, 0.24206905],
              [0.3113967 , 0.22512221, 0.23296535]]],
     
     
            [[[0.30064338, 0.4026042 , 0.46927086],
              [0.41646537, 0.50049895, 0.5672444 ],
              [0.15277047, 0.22728029, 0.2849615 ],
              ...,
              [0.6408222 , 0.44952774, 0.38970184],
              [0.5514177 , 0.3993255 , 0.33012912],
              [0.6118902 , 0.35699135, 0.32748792]],
     
             [[0.07331932, 0.1331889 , 0.195934  ],
              [0.21690302, 0.30204394, 0.36394435],
              [0.10522589, 0.1797357 , 0.23553489],
              ...,
              [0.7057689 , 0.47687414, 0.43616158],
              [0.6470366 , 0.49107108, 0.43701383],
              [0.64858764, 0.40256503, 0.38597727]],
     
             [[0.24465162, 0.33092615, 0.38486084],
              [0.21227679, 0.2893076 , 0.34624913],
              [0.28402928, 0.35920873, 0.41210175],
              ...,
              [0.78919715, 0.5308431 , 0.5044468 ],
              [0.644393  , 0.5003805 , 0.45974243],
              [0.64082295, 0.41336197, 0.4073269 ]],
     
             ...,
     
             [[0.35639447, 0.35198268, 0.29291406],
              [0.1667936 , 0.16116072, 0.11650035],
              [0.16399686, 0.15797883, 0.13037467],
              ...,
              [0.43435144, 0.5155842 , 0.23250015],
              [0.38325384, 0.48997656, 0.16980837],
              [0.26687855, 0.39685088, 0.05557318]],
     
             [[0.15662642, 0.19522935, 0.14817053],
              [0.14003852, 0.16769521, 0.13433562],
              [0.14999563, 0.16376051, 0.15150999],
              ...,
              [0.46663687, 0.54948455, 0.2580847 ],
              [0.38925105, 0.49597377, 0.17580557],
              [0.29099146, 0.4209638 , 0.09616793]],
     
             [[0.1804622 , 0.23573181, 0.2047269 ],
              [0.16674107, 0.2080051 , 0.18802084],
              [0.17789304, 0.20785628, 0.20215775],
              ...,
              [0.58226466, 0.667419  , 0.36909926],
              [0.3686067 , 0.47841063, 0.14815848],
              [0.4694088 , 0.59938115, 0.24193154]]]], dtype=float32),
     array([[False, False, False, ..., False, False, False],
            [False, False, False, ..., False, False, False],
            [False, False, False, ..., False, False, False],
            ...,
            [False, False, False, ..., False, False, False],
            [False, False, False, ..., False, False, False],
            [False, False, False, ..., False, False, False]]))




```python
len(train_images),len(train_labels)
```




    (32, 32)




```python
# visualize the data
train_images,train_labels=next(train_data.as_numpy_iterator())
show_25_images(train_images,train_labels)
```


    
![png](output_55_0.png)
    



```python
val_images,val_labels=next(val_data.as_numpy_iterator())
show_25_images(val_images,val_labels)
```


    
![png](output_56_0.png)
    


## Building a model

Before building a model, there are a few things we need to define:

* The input shape (our image shape, in the from of Tensors) of our model.
* The output shape (image labels, in the form of Tensors) of our model.
* The URL of the model we want to use from TensorFlow Hub - https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4



```python
IMG_SIZE
```




    224




```python
# setup input shape to the model
INPUT_SHAPE=[None,IMG_SIZE,IMG_SIZE,3]    # [batch,height,width,color channels]

# setup output shape of our model
OUTPUT_SHAPE=len(unique_breeds)

# setup model URL from TensorFlow Hub
MODEL_URL="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
```

Put inputs, outputs and model together into a `Keras` deep learning model.

Create a function:
* Take the input shape, output shape and the model as parameters.
* Define the layers in a Keras model in **sequential** fashion.
* Compile the model.
* Build the model.
* Return the model.

https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn


```python
def create_model(input_shape=INPUT_SHAPE,output_shape=OUTPUT_SHAPE,model_url=MODEL_URL):
  print("Building model with:",MODEL_URL)

  # setup model layers
  model=tf.keras.Sequential([
                             hub.KerasLayer(MODEL_URL), # layer 1 (input)
                             tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                                                   activation="softmax")  # layer 2 (output)
  ])

  # compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=['accuracy']
  )

  # build the model
  model.build(INPUT_SHAPE)

  return model
```


```python
model=create_model()
model.summary()
```

    Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    keras_layer_2 (KerasLayer)   (None, 1001)              5432713   
    _________________________________________________________________
    dense_2 (Dense)              (None, 120)               120240    
    =================================================================
    Total params: 5,552,953
    Trainable params: 120,240
    Non-trainable params: 5,432,713
    _________________________________________________________________



```python
outputs=np.ones(shape=(1,1,1280))
outputs
```




    array([[[1., 1., 1., ..., 1., 1., 1.]]])



## Creating callbacks

Callbacks are helper functions a model can use during training to do such things as save its progress, check its progress or strop training early if a model stops improving.

Create two callbacks, one for `TensorBoard` which helps track our model progress and another for early stopping which prevents our model from training too long.

### TensorBoard Callback

1. Load TensorBoard notebook.
2. Create a TensorBoard callback which is able to save logs to a directory and pass it to our model's `fit()` cunction.
3. Visualize model's training logs with the
`%tensorboard` magic function.


```python
# load TensorBoard notebook extension
%load_ext tensorboard
```


```python
import datetime

# create a function to build a TensorBoard callback
def create_tensorboard_callback():
  # create a log direcotry for storing logs
  logdir=os.path.join('/content/drive/MyDrive/Colab Notebooks/Dog Vision/logs',
                      # make it so the logs get tracked whenever running an experiment
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                     )
  return tf.keras.callbacks.TensorBoard(logdir)
```

### Early stopping callback

Early stopping helps our model from **overfitting** by stopping training if a certain evaluation metric stops improving. 


```python
early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                patience=3)
```

## Training a model (on subset of data)

Our first model is only going to train on 1000 images, to make sure it's working.


```python
NUM_EPOCHS=100 #@param {type:'slider",min:10,max:100,step:10}
```


```python
# check GPU
print("GPU","available (YES)"  if tf.config.list_physical_devices('GPU') else "not available")
```

    GPU available (YES)


Create a function to train a model.

* Create a model using `create_model()`
* Setup a TensorBoard callback using `create_tensorboard_callback()`
* Call the `fit()` function on the model passing it the training data, validation data, number of epochs to train for (`NUM_EPOCHS`) and the callbacks
* Return the model


```python
# build a function to train and return a trainded model

def train_model():
  """
  Trains a given model and returns the trained version.
  """
  # create a model
  model=create_model()

  # create new TensorBoard session everytime we train a model
  tensorboard=create_tensorboard_callback()

  # fit the model to the data passing it the callbacks we created
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard,early_stopping])
  
  return model
```


```python
# fit the model
model=train_model()
```

    Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
    Epoch 1/100
    25/25 [==============================] - 213s 8s/step - loss: 5.2834 - accuracy: 0.0397 - val_loss: 3.5002 - val_accuracy: 0.2400
    Epoch 2/100
    25/25 [==============================] - 5s 183ms/step - loss: 1.9969 - accuracy: 0.6126 - val_loss: 2.1960 - val_accuracy: 0.4900
    Epoch 3/100
    25/25 [==============================] - 5s 182ms/step - loss: 0.6745 - accuracy: 0.8996 - val_loss: 1.6466 - val_accuracy: 0.6100
    Epoch 4/100
    25/25 [==============================] - 5s 183ms/step - loss: 0.2909 - accuracy: 0.9874 - val_loss: 1.4662 - val_accuracy: 0.6250
    Epoch 5/100
    25/25 [==============================] - 5s 183ms/step - loss: 0.1615 - accuracy: 0.9973 - val_loss: 1.3867 - val_accuracy: 0.6450
    Epoch 6/100
    25/25 [==============================] - 5s 185ms/step - loss: 0.1061 - accuracy: 0.9966 - val_loss: 1.3533 - val_accuracy: 0.6500
    Epoch 7/100
    25/25 [==============================] - 5s 183ms/step - loss: 0.0780 - accuracy: 1.0000 - val_loss: 1.3152 - val_accuracy: 0.6600
    Epoch 8/100
    25/25 [==============================] - 5s 183ms/step - loss: 0.0643 - accuracy: 1.0000 - val_loss: 1.2943 - val_accuracy: 0.6550
    Epoch 9/100
    25/25 [==============================] - 5s 187ms/step - loss: 0.0493 - accuracy: 1.0000 - val_loss: 1.2734 - val_accuracy: 0.6650
    Epoch 10/100
    25/25 [==============================] - 5s 184ms/step - loss: 0.0420 - accuracy: 1.0000 - val_loss: 1.2609 - val_accuracy: 0.6600
    Epoch 11/100
    25/25 [==============================] - 5s 182ms/step - loss: 0.0359 - accuracy: 1.0000 - val_loss: 1.2439 - val_accuracy: 0.6650
    Epoch 12/100
    25/25 [==============================] - 5s 183ms/step - loss: 0.0310 - accuracy: 1.0000 - val_loss: 1.2336 - val_accuracy: 0.6650


It looks like the model is overfitting because it's performing far better on the training dataset (1.0000) than the validation dataset (0.6650).
Overfitting to begin with is a good thing.

### Checking the TensorBoard logs

The TensorBoard magic function (`%tensorboard`) will access the logs directory and visualize its contents.


```python
%tensorboard --logdir drive/MyDrive/Colab\ Notebooks/Dog\ Vision/logs
```


    <IPython.core.display.Javascript object>


## Making and evaluating predictions using a trained model


```python
val_data
```




    <BatchDataset shapes: ((None, 224, 224, 3), (None, 120)), types: (tf.float32, tf.bool)>




```python
# make predictions on the validation data (not used to train on)
predictions=model.predict(val_data,verbose=1)
predictions
```

    7/7 [==============================] - 1s 124ms/step





    array([[2.96485639e-04, 4.61363197e-05, 5.52320853e-04, ...,
            1.46665180e-03, 1.01416890e-05, 1.65006681e-03],
           [9.58894659e-03, 1.82944292e-03, 1.20498976e-02, ...,
            1.35627086e-03, 2.87414598e-03, 1.22991711e-04],
           [5.82929033e-06, 1.58805688e-05, 1.76763278e-05, ...,
            7.18082156e-05, 1.06802268e-04, 3.78410285e-03],
           ...,
           [2.11677252e-06, 2.06057121e-05, 1.36807785e-05, ...,
            4.56108774e-06, 1.41118235e-05, 1.89351740e-05],
           [1.86676590e-03, 9.22191713e-04, 1.41811499e-04, ...,
            3.49212438e-04, 3.60381091e-05, 8.68038647e-03],
           [4.67393606e-04, 4.05903011e-05, 1.37729337e-03, ...,
            6.47077337e-03, 2.64649483e-04, 3.09605239e-04]], dtype=float32)




```python
len(predictions[0])
```




    120




```python
predictions.shape
```




    (200, 120)




```python
len(y_val),len(unique_breeds)
```




    (200, 120)




```python
np.max(predictions[0])
```




    0.19074444




```python
np.sum(predictions[0])
```




    0.99999994




```python
# first prediction
index=42
print(predictions[index])
print(f"max value (probability of prediction): {np.max(predictions[index])}")
print(f"sum: {np.sum(predictions[index])}")
print(f"max index: {np.argmax(predictions[index])}")
print(f"predicted label: {unique_breeds[np.argmax(predictions[index])]}")
```

    [2.98793184e-05 7.11702596e-05 4.04683014e-05 5.69886288e-05
     1.68637454e-03 4.79032715e-05 3.39595223e-04 8.66208051e-04
     8.08137842e-03 3.95613089e-02 2.05670367e-05 1.34134698e-05
     3.71900416e-04 7.28554465e-03 1.13078207e-03 4.22544824e-03
     1.52088478e-05 1.02701008e-04 5.10835729e-04 3.63791158e-04
     4.05551873e-05 6.69662782e-04 2.27022811e-05 2.77535728e-04
     6.84955530e-03 1.17226286e-04 3.35554760e-05 6.94397022e-05
     1.72652624e-04 2.04530006e-05 2.17301003e-05 1.32953530e-04
     1.70780680e-04 7.15565475e-05 2.61657751e-05 1.95891516e-05
     8.56089900e-05 3.98193777e-04 7.79896582e-05 2.57055014e-01
     2.06472672e-04 2.02790943e-05 8.65258183e-03 3.11668846e-05
     2.93666264e-04 3.58575744e-05 3.11292788e-05 5.74399193e-04
     8.53073579e-05 6.80369267e-04 9.05373454e-05 1.38944510e-04
     1.49340049e-04 1.86859863e-03 5.06994511e-05 4.50387452e-04
     8.27256372e-05 1.45079703e-05 5.09739948e-05 1.33656740e-05
     4.70449304e-05 4.96894238e-04 2.12170089e-05 4.52871318e-05
     2.69783759e-05 4.35864466e-04 6.38550409e-05 1.35875645e-03
     2.93962716e-04 6.85716950e-05 5.27507145e-05 3.19051942e-05
     1.16234849e-04 1.11734215e-03 1.06418120e-04 1.13028451e-04
     2.54220766e-04 1.00490477e-04 3.90628156e-05 5.44033886e-04
     9.99931217e-06 1.15649033e-04 2.54268034e-05 6.95931783e-04
     5.53756021e-04 2.83925510e-05 3.94323142e-05 3.28315691e-05
     8.60372529e-06 7.82665331e-04 1.45860278e-04 2.94291858e-05
     2.23804120e-04 6.73382310e-04 2.94304500e-05 7.10672975e-05
     1.51038239e-05 4.65082376e-05 1.14582017e-05 2.71994650e-04
     1.20128971e-04 7.74617438e-05 1.11090121e-04 1.25613093e-04
     1.12496768e-04 5.17163680e-05 5.55073275e-05 3.07158043e-05
     2.31169943e-05 3.71850037e-04 4.88710975e-05 2.00852845e-03
     2.79575121e-04 6.37095511e-01 3.83623468e-04 7.30030704e-04
     2.28671397e-05 8.31052748e-05 4.24118713e-03 2.06711848e-04]
    max value (probability of prediction): 0.6370955109596252
    sum: 1.0000001192092896
    max index: 113
    predicted label: walker_hound



```python
# turn prediction probabilities into the respective label
def get_pred_label(prediction_probabilities):
  """
  Turn an array of prediction probabilities into a label
  """
  return unique_breeds[np.argmax(prediction_probabilities)]

# eg
pred_label=get_pred_label(predictions[81])
pred_label
```




    'dingo'



Since the validation data is still in a batch dataset, we'll have to unbachify it to make predictions on the validation images and then compare those predictions to the validation labels (true labels).


```python
val_data
```




    <BatchDataset shapes: ((None, 224, 224, 3), (None, 120)), types: (tf.float32, tf.bool)>




```python
# unbatch a batch dataset
def unbatchify(data):
  """
  Take a batched dataset of (image,label) Tensors and return separate arrays
  of images and labels.
  """
  images=[]
  labels=[]

  for image,label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])

  return images,labels

val_images,val_labels=unbatchify(val_data)
val_images[0],val_labels[0]
```




    (array([[[0.29599646, 0.43284872, 0.3056691 ],
             [0.26635826, 0.32996926, 0.22846507],
             [0.31428418, 0.2770141 , 0.22934894],
             ...,
             [0.77614343, 0.82320225, 0.8101595 ],
             [0.81291157, 0.8285351 , 0.8406944 ],
             [0.8209297 , 0.8263737 , 0.8423668 ]],
     
            [[0.2344871 , 0.31603682, 0.19543913],
             [0.3414841 , 0.36560842, 0.27241898],
             [0.45016077, 0.40117094, 0.33964607],
             ...,
             [0.7663987 , 0.8134138 , 0.81350833],
             [0.7304248 , 0.75012016, 0.76590735],
             [0.74518913, 0.76002574, 0.7830809 ]],
     
            [[0.30157745, 0.3082587 , 0.21018331],
             [0.2905954 , 0.27066195, 0.18401104],
             [0.4138316 , 0.36170745, 0.2964005 ],
             ...,
             [0.79871625, 0.8418535 , 0.8606443 ],
             [0.7957738 , 0.82859945, 0.8605655 ],
             [0.75181633, 0.77904975, 0.8155256 ]],
     
            ...,
     
            [[0.9746779 , 0.9878955 , 0.9342279 ],
             [0.99153054, 0.99772066, 0.9427856 ],
             [0.98925114, 0.9792082 , 0.9137934 ],
             ...,
             [0.0987601 , 0.0987601 , 0.0987601 ],
             [0.05703771, 0.05703771, 0.05703771],
             [0.03600177, 0.03600177, 0.03600177]],
     
            [[0.98197854, 0.9820659 , 0.9379411 ],
             [0.9811992 , 0.97015417, 0.9125648 ],
             [0.9722316 , 0.93666023, 0.8697186 ],
             ...,
             [0.09682598, 0.09682598, 0.09682598],
             [0.07196062, 0.07196062, 0.07196062],
             [0.0361607 , 0.0361607 , 0.0361607 ]],
     
            [[0.97279435, 0.9545954 , 0.92389745],
             [0.963602  , 0.93199134, 0.88407487],
             [0.9627158 , 0.9125331 , 0.8460338 ],
             ...,
             [0.08394483, 0.08394483, 0.08394483],
             [0.0886985 , 0.0886985 , 0.0886985 ],
             [0.04514172, 0.04514172, 0.04514172]]], dtype=float32), 'cairn')




```python
get_pred_label(val_labels[0])
```




    'cairn'




```python
images_=[]
labels_=[]

# loop through unbatched data
for image,label in val_data.unbatch().as_numpy_iterator():
  images_.append(image)
  labels_.append(label)

images_[0],labels_[0]
```




    (array([[[0.29599646, 0.43284872, 0.3056691 ],
             [0.26635826, 0.32996926, 0.22846507],
             [0.31428418, 0.2770141 , 0.22934894],
             ...,
             [0.77614343, 0.82320225, 0.8101595 ],
             [0.81291157, 0.8285351 , 0.8406944 ],
             [0.8209297 , 0.8263737 , 0.8423668 ]],
     
            [[0.2344871 , 0.31603682, 0.19543913],
             [0.3414841 , 0.36560842, 0.27241898],
             [0.45016077, 0.40117094, 0.33964607],
             ...,
             [0.7663987 , 0.8134138 , 0.81350833],
             [0.7304248 , 0.75012016, 0.76590735],
             [0.74518913, 0.76002574, 0.7830809 ]],
     
            [[0.30157745, 0.3082587 , 0.21018331],
             [0.2905954 , 0.27066195, 0.18401104],
             [0.4138316 , 0.36170745, 0.2964005 ],
             ...,
             [0.79871625, 0.8418535 , 0.8606443 ],
             [0.7957738 , 0.82859945, 0.8605655 ],
             [0.75181633, 0.77904975, 0.8155256 ]],
     
            ...,
     
            [[0.9746779 , 0.9878955 , 0.9342279 ],
             [0.99153054, 0.99772066, 0.9427856 ],
             [0.98925114, 0.9792082 , 0.9137934 ],
             ...,
             [0.0987601 , 0.0987601 , 0.0987601 ],
             [0.05703771, 0.05703771, 0.05703771],
             [0.03600177, 0.03600177, 0.03600177]],
     
            [[0.98197854, 0.9820659 , 0.9379411 ],
             [0.9811992 , 0.97015417, 0.9125648 ],
             [0.9722316 , 0.93666023, 0.8697186 ],
             ...,
             [0.09682598, 0.09682598, 0.09682598],
             [0.07196062, 0.07196062, 0.07196062],
             [0.0361607 , 0.0361607 , 0.0361607 ]],
     
            [[0.97279435, 0.9545954 , 0.92389745],
             [0.963602  , 0.93199134, 0.88407487],
             [0.9627158 , 0.9125331 , 0.8460338 ],
             ...,
             [0.08394483, 0.08394483, 0.08394483],
             [0.0886985 , 0.0886985 , 0.0886985 ],
             [0.04514172, 0.04514172, 0.04514172]]], dtype=float32),
     array([False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False,  True,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False,
            False, False, False]))




```python
get_pred_label(labels_[0])
```




    'cairn'




```python
get_pred_label(predictions[0])
```




    'irish_wolfhound'



Visualize:

* Prediction labels
* Validation labels (truth labels)
* Validation images

Create a function:

* Take an array of prediction probabilities, an array of truth labels and an array of images and an integer.
* Convert the prediction probabilities to a predicted label.
* Plot the predicted label, its predicted probability, the truth label and the target image on a single plot.


```python
def plot_pred(prediction_probabilities,labels,images,n=1):
  """
  View the prediction, ground truth and image for sample n
  """
  pred_prob,true_label,image=prediction_probabilities[n],labels[n],images[n]

  # get the pred label
  pred_label=get_pred_label(pred_prob)

  # plot 
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  if pred_label==true_label:
    color='green'
  else:
    color='red'

  plt.title("{} {:2.0f}% {}".format(pred_label,
                                    np.max(pred_prob)*100,
                                    true_label),
            color=color
            )
 


```


```python
plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images)
```


    
![png](output_97_0.png)
    



```python
plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images,n=42)
```


    
![png](output_98_0.png)
    



```python
plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images,n=77)
```


    
![png](output_99_0.png)
    


Make a function to view the model's top 10 predictions.

The function:
* Take an imput of prediction probabilities array and a ground truth array and an integer
* Find the prediction using `get_pred_label()`
* Find the top 10:
   * Prediction probabilities indexes
   * Prediction probabilities values
   * Prediction labels
* Plot the top 10 prediction probability values and labels, coloring the true label green.


```python
def plot_pred_conf(prediction_probabilities,labels,n=1):
  """
  Plot the top 10 highest prediction confidences along with the truth label for sample n.
  """

  pred_prob,true_label=prediction_probabilities[n],labels[n]

  pred_label=get_pred_label(pred_prob)

  # find the top 10
  top_10_pred_indexes=pred_prob.argsort()[-10:][::-1]
  top_10_pred_values=pred_prob[top_10_pred_indexes]
  top_10_pred_labels=unique_breeds[top_10_pred_indexes]
  
  # plot
  top_plot=plt.bar(np.arange(len(top_10_pred_labels)),
                   top_10_pred_values,
                   color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation='vertical')
  
  if np.isin(true_label,top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels==true_label)].set_color('green')
  else:
    pass

  #print(top_10_pred_labels)
  #print(true_label)

```


```python
predictions[0]
```




    array([2.96485639e-04, 4.61363197e-05, 5.52320853e-04, 3.80371530e-05,
           5.00787792e-05, 4.17037745e-06, 1.69642810e-02, 1.19220163e-03,
           6.57710625e-05, 3.47458204e-04, 3.02767701e-04, 5.44649265e-05,
           1.58896233e-04, 1.54315596e-04, 2.12866697e-04, 6.81604259e-04,
           7.55681904e-05, 3.25117446e-02, 8.17434193e-06, 1.04332885e-05,
           5.95758669e-04, 2.66427436e-04, 3.56164128e-05, 3.40718776e-03,
           1.22535985e-05, 7.30119427e-05, 6.70529827e-02, 1.48341962e-04,
           9.83689213e-04, 5.35331565e-05, 1.21229328e-04, 1.06281263e-03,
           7.57041911e-04, 6.29976930e-05, 4.24430269e-04, 4.89092767e-02,
           1.53772853e-05, 2.43704329e-04, 3.89078850e-05, 3.57579469e-04,
           4.41184128e-03, 8.20340501e-05, 3.05807276e-04, 1.16355950e-04,
           1.27658450e-05, 4.48119281e-05, 1.44995884e-05, 9.29785601e-05,
           1.28638043e-04, 5.79067506e-04, 1.36438088e-04, 9.35194748e-06,
           8.55011691e-04, 3.37218153e-05, 2.34732215e-04, 1.90215414e-05,
           1.59449773e-04, 7.99123850e-03, 1.43519323e-03, 1.90744445e-01,
           4.68069367e-04, 5.20525973e-05, 6.92081812e-04, 2.40851423e-05,
           1.77740629e-04, 4.99222381e-03, 5.52333295e-05, 1.05029566e-03,
           1.44749898e-02, 3.21431842e-04, 9.10677165e-02, 3.75569914e-04,
           2.40816866e-04, 8.61005858e-03, 3.36936238e-04, 2.16002645e-05,
           7.54338782e-03, 3.83607708e-02, 1.40515927e-04, 1.12398500e-02,
           7.56609952e-04, 7.36765796e-03, 1.78233473e-04, 2.47257240e-02,
           2.18739297e-04, 3.28396331e-04, 3.37568374e-04, 5.86047303e-04,
           4.85639728e-04, 8.55967926e-04, 7.81782903e-04, 5.64997499e-05,
           2.03699419e-05, 2.72517675e-03, 2.14720945e-04, 9.75599978e-05,
           1.39346463e-03, 1.15599914e-03, 1.33485737e-04, 2.78196621e-05,
           3.06276605e-02, 9.61534170e-05, 7.96294734e-02, 7.96772540e-02,
           4.18008916e-04, 2.08212645e-04, 1.59320999e-02, 2.68195108e-05,
           3.77106044e-05, 1.78109616e-01, 1.21858560e-04, 1.47853701e-04,
           1.00870027e-04, 7.40224423e-05, 3.65641579e-04, 5.80180458e-05,
           2.79157492e-03, 1.46665180e-03, 1.01416890e-05, 1.65006681e-03],
          dtype=float32)




```python
predictions[0].argsort()[-10:][::-1]
```




    array([ 59, 109,  70, 103, 102,  26,  35,  77,  17, 100])




```python
predictions[0][predictions[0].argsort()[-10:][::-1]]
```




    array([0.19074444, 0.17810962, 0.09106772, 0.07967725, 0.07962947,
           0.06705298, 0.04890928, 0.03836077, 0.03251174, 0.03062766],
          dtype=float32)




```python
predictions[0].max()
```




    0.19074444




```python
unique_breeds[predictions[0].argsort()[-10:][::-1]]
```




    array(['irish_wolfhound', 'tibetan_terrier', 'lhasa',
           'soft-coated_wheaten_terrier', 'silky_terrier', 'cairn',
           'dandie_dinmont', 'miniature_schnauzer', 'border_terrier',
           'shih-tzu'], dtype=object)




```python
plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)
```


    
![png](output_107_0.png)
    



```python
plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=1)
```


    
![png](output_108_0.png)
    



```python
plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=77)
```


    
![png](output_109_0.png)
    



```python
# check out a few predictions and their different values
i_multiplier=20
num_rows=3
num_cols=2
num_images=num_rows*num_cols
plt.figure(figsize=(10*num_cols,5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows,2*num_cols,2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows,2*num_cols,2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                 labels=val_labels,
                 n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()
```


    
![png](output_110_0.png)
    


## Saving and reloading a trained model


```python
# create a function to save a model
def save_model(model,suffix=None):
  """
  Save a given model and appends a suffix (string).
  """
  # create a model directory pathname with current time
  modeldir=os.path.join("/content/drive/MyDrive/Colab Notebooks/Dog Vision/models",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
  model_path=modeldir+'-'+suffix+'.h5'
  print(f"saving model to: {model_path}...")
  model.save(model_path)
  return model_path
```


```python
# create a function to load a model
def load_model(model_path):
  """
  Load a saved model from a specified path
  """
  print(f"loading saved model from: {model_path}")
  model=tf.keras.models.load_model(model_path,
                                   custom_objects={"KerasLayer":hub.KerasLayer})
  
  return model
```


```python
# save our model trained on 1000 images
save_model(model,suffix="1000-images-mobilenetv2-Adam")
```

    saving model to: /content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-11351614339311-1000-images-mobilenetv2-Adam.h5...





    '/content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-11351614339311-1000-images-mobilenetv2-Adam.h5'




```python
# load a trained model
loaded_1000_image_model=load_model('/content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-11351614339311-1000-images-mobilenetv2-Adam.h5')
```

    loading saved model from: /content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-11351614339311-1000-images-mobilenetv2-Adam.h5



```python
# evaluate the pre-saved model
model.evaluate(val_data)
```

    7/7 [==============================] - 1s 108ms/step - loss: 1.2336 - accuracy: 0.6650





    [1.233637809753418, 0.6650000214576721]




```python
# evaluate the loaded model
loaded_1000_image_model.evaluate(val_data)
```

    7/7 [==============================] - 1s 106ms/step - loss: 1.2336 - accuracy: 0.6650





    [1.233637809753418, 0.6650000214576721]



## Training a model (on the full data)


```python
len(X),len(y)
```




    (10222, 10222)




```python
X[:10]
```




    ['/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/000bec180eb18c7604dcecc8fe0dba07.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001cdf01b096e06d78e9e5112d419397.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00214f311d5d2247d5dfe4fe24b2303d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0021f9ceb3235effd7fcde7f7538ed62.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002211c81b498ef88e1b40b9abf84e1d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00290d3e1fdd27226ba27a8ce248ce85.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002a283a315af96eaea0e28e7163b21b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/003df8b8a8b05244b1d920bb6cf451f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0042188c895a2f14ef64a918ed9c7b64.jpg']




```python
X_train
```




    ['/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00bee065dcec471f26394855c5c2f3de.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d2f9e12a2611d911d91a339074c8154.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1108e48ce3e2d7d7fb527ae6e40ab486.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dc3196b4213a2733d7f4bdcd41699d3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/146fbfac6b5b1f0de83a5d0c1b473377.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ea5759640f2e1c2d1a06adaf8a54ca7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03e1d2ee5fd90aef036c90a9e7f81177.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16941a6728ddb9cb7423a6cc97fbe071.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bedbecd92390ef9f4f7c8b06a629340.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/143b9484273e57668d03bfc26755810a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/019ff93e03802e661577b5869e099dcb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/035f28d8ad34afaf7c8d276d6674bf8f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16ac25747d1a51db033d6461156ddb0b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14597c96d5c222eebd742f4207296314.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ff07d44f992eec7f0b5452875255c80.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07f95d08b40f1402abdab79b1f834e4c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/102fe645c59f482cbc771c01cfff3ff9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05c128c8e3ef0c2739f181f9c5677f56.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1639ccef237eec9fc4eb1aa0916201c3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/117085659f91228627caf21a574f2bbf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/075a28044780636f48d8571f1d32f73d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1234dd4303d5181574ca007f53ed03db.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17f69b6a170c33a8786d566f6dc9b8d5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0140b05bfc2fd43f2819fab3d8566109.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15d7f26c7d32f81c204936362ead551e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001cdf01b096e06d78e9e5112d419397.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d693ab130c18e15f923f59eb102def9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0861355ea326a82de3aab420d4276e5b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16265434d841173019be215bea5d8097.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d303784e6740e5de249e1f2078a7b4b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/084c4f43ecc2630587de6c3e543525c3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/187b3a867bb68860208c37aaf43d2115.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b86485ef980b0b4999e7ede1f90999e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e24a584e10db2c8f827ab00772e29cf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00e9ed3fab1d2032603d1a90e557976f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00cc68a50b2d016a6b29af628ea4e04b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0566a60d7504a6fad4161d0ef2765a34.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17a64c7a240d8c3b208eb371297189a6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/046d8f04a5a42872774110c6a2db1224.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08098e0a4074f62169ab53f7efe40da2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00214f311d5d2247d5dfe4fe24b2303d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04cce0cd53c6f01d242a49e43de513a1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c68a578981993b919e89c611f04a97f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13cab1309bdaf21aa17cf71fd6f88a65.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0769e276e8b9b992a3fcd6a10deee6a7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/029ed6af5dcadfd105db9271e12ede14.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a5f744c5077ad8f8d580081ba599ff5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01e2245b46eb747260ff80f1c892daef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0206c12e8984e3c8a166cc272de25d6f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06e548623971fbdf8c80b0614f811f50.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01f429667104c0c5a5f321700f15435c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/185dd9ac0589562442f553cf686783d4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13932827f30cb00d0cb4c40443c788f4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15d17b41ac5230af8f31417f0a673915.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09927a82d649607a7704ac6043cdfa9a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07d6e906eb95f24c53eb92ecc2d47783.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c9fc0f58a6724ca680c76785a452bba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1690522901e3d41a84c3dc91b1520902.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16a6a63387fa1f0b4a0dcf6bb05f6204.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002211c81b498ef88e1b40b9abf84e1d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/120c338a78b113785c31e4ebf11d229e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b1e71a499a26eaacc06991bca982523.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02508e76981e1ba059d785704b4c480c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/064dff92e5654a8679f8971a027a4040.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08829c00da02dea80eb491122989492f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/152aa0851b6a1349b99044f16eadc59d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13c7c6b47b97b9f0a4591fef29893436.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/093fc67079701f48bd1d2e52a684ed9e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f7c817dd5f8b8d6b57e3b7f3e2f4f56.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09438173067ccee8d4a1a45f2f8a8eb6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bbb9ce1f03a205fead338f0be3040bc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02d54f0dfb40038765e838459ae8c956.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16052ac2a6ff7f1fbbc85885d2a7c467.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05cab78114abb08afcda78ee70222edd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0cb5839c82f7c1286e55f260d4e608c4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0042188c895a2f14ef64a918ed9c7b64.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0490b67cb414d527d6c21052b1e3b5dd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0eb6e0f872b393654bea530bfc2cac1d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01f397f16fb2d7d76fdbce4e2207c1a3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/137f07bc5f15fa8e39f85a8333e68780.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e7691c13a9267b621ffb866a83d08aa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13c630fcf8aa68ab0e97cd4a644b8943.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05934390044791b8fd2dd2ba9c0b4b36.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07096719a671e1737b829bd1a88f1dbf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04dea47b883acca9bd57c7ea2930524f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11a508588bc3dec81bd4ce4913f963cd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/088be435c5245f79c448812169a30cb5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0aa31d6ecdb95ac7d3b7fa42b86bd91b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0543510e763cbf8fc771a097ffda9984.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17324f3d13ca421725028de23d631e03.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17193d29b9833d783133f4b13f12f513.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13c5a0d636619210b5ac003ee82aacf0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c19867277e6c96ad8f487b4fe343ff9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b1cdb2a36dd432fcc7567959cb23798.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08946697b3e50e602d6bea765c8fb9f5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b30d48dd2818cdea768d884623e8c2a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16e2c529e31eb3d2a276eb5bda2b009f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a541df84dd3febb076e2c33a23cd230.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/002a283a315af96eaea0e28e7163b21b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03bda5c85206a273eb978c7b9483df5f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d579f92d60935a54aa6dba6e6213257.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a7f8d31c960071d0d4cfba36c0159c7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b97116ed04c8f0f7eb4a2b4b2620476.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a27d304c96918d440e79e6e9e245c3f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1352efb02c29ba1b9be918170afff486.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/172fcb0b8bbd67401c9e53e5f9bf39bc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/080ebefee69f27243b185c113052963e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e33a6b9235d5cbb8712daf0abbd0007.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16df5020c2b6ac3f70fb3a5c27c5175e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0df3cd4042fa7c2a6e008a3a849a23d1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e96d18ac522425bd231c3d8a32159da.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d0d0f7c689020c35b83a91e7717624b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e549376cfed70cf2a5b84e7a42d85ef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1165dd8264b3998810065ab9e3cecb3b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a9f8f6543d0b33fe9474035dd5323ef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0abdda879bb143b19e3c480279541915.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/054f069e01a809b23a9da31dd8f4841e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/086e8ca78ec3303a8f06df003ecb6612.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11f1e40f11ca1babbcc547c5c98a1a3c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04e370c8510638ca969a716822444085.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01e787576c003930f96c966f9c3e1d44.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/039bdddd8546f0fafdf984b810d1138b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0743bfe9525b7a6310a297b11b7e154a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f73c1cc99dd8b9c580c4260debf1f93.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1351c4f7cec88e8ba56ada4c772d84f5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c286c574ef09562c123dafe5ce6ac23.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0825e8471e3c9a14dc341bdd8630f05e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b9fc8956444d4f5296ace3a4928541b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1177d2702335c889d4c1456e45e3d2ef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/032620ae0f847d957d94d1fd76cb17e8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11476f28c0143a77c536b597a177abc9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/079509ceab0968ce07cffa0df479f1b0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11c3b389906a9302def1e873c9f9b6c9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f74df5d1c925541eb1b3031a82a5cb8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/000bec180eb18c7604dcecc8fe0dba07.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07a215abb3ff16fd19a5b832c8f3408b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/115ad26b0cef153d2afaf6985503c9c3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d6c6238a4cc499bb57fa0c10a15f8ad.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/153697b802875722cb25421a661a4526.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d1d7bacd20ef921e9b3fce6d9b9f9f0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b6837976df682360dc9ef25b81bc893.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09a6fd935dd1a4dee29b97885cfa9318.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/040fef64640b89c53f161b6c5215b78b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02f3d5fd1eea0a2e6f4742bc54b51ba9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/098c50d7ba4f07d9aba8c6162aa70b50.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a1b0b7df2918d543347050ad8b16051.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d8d386390ba269dd4e475c20b91e8f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/160f932c413228553024d4cc9bafc156.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/022b34fd8734b39995a9f38a4f3e7b6b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16fd2ee2057012b1d3db46e219a4d022.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0440acb104d7346ccc9bba0716603f6a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/068314c65cbe67fa4f57283f4f3a801c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15d25fb9d0b922b4943b312509e95c21.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01b35a06b00e4a832b935cf8d51303c1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0709abee3095717f43db409dd7ce769b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/031d3353588a81b42bcae74ff19571b0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/073bb2f975ee39406e692e33364762a3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a409042dd36b2c8293c67d4d4ea9ef9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07e103146c1f4513647ddb9614210320.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0365fe4e3e13a885c6b02ebbbf2d9173.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a3f1898556115d6d0931294876cd1d9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e532335572b91360975154c90381689.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0326d40cc2b35521f1f3e73cabcf2a23.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ab12906ac3a87b531c574a15e79c58a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03dfefdb2939388f7ca9578cb7c4a2b9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00ba244566e36e0af3d979320fd3017f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/131d708a270a3ebca285978bf53df62e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/047d9a7dc7205c76e5487d6fc5f4a666.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0964a2558fc9aa293a6a934d49f64968.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14c8f5fc68ca8fb6b33a0082849b15ad.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03f2f64aebc483ef8e5e17aba7311aa6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1158819241bb6eee2c1ff790a885ecf3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/035b459eb52acba2ae75bed041b368fe.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/041a54577da744348d4da8094e882cd9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00290d3e1fdd27226ba27a8ce248ce85.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12815630b892efa2744926fe59382491.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b6c5bc46b7a0e29cddfa45b0b786d09.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01c230ec18eed427cf5b1db1a833024d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/164c3b823c400cea20ad0a3426e1eeea.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/141321373d0535ee37171a2d6ca9bcbb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/058ea5b467d08ed978658dc1ad85fb7d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0374e657c8b3b55579751adcaa7c8023.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/139e8120addf833f72a8dc2c12ad4c8a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/056c314f2a7d119447af259a07eb31df.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/050073020e80e4935b53df8d786c8612.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16a760e466fd2b2fe0ef8879b5b2245d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c0fb87fb13f0f6c8d21ff6c15e3361e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0325e6ba5b6afd3e0dd94c3a77b6de59.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1857ac9d2306fc646d4bb817aaa20424.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1321d074e879267699d42fdca77b4004.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/044efbf8ab3e7195b8ea3c92ef0d48df.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/011bd7fd7c036dbb243d0e37329c77b7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1116f9aa5db9ee09090115d09f327093.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0296633efc21006e3ce2af7eaeff04f7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e7712450813da57f7df73fc45a183b4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0162107acd8f2588c0944b791d61bb0c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09aec2a169942b17d88b4b5f1bba5886.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a896d2b3af617df543787b571e439d8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12c9f56e73ac9f4a08afb142f89051ae.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/189cc11eaaa9152fffb49e4fa5b14ebd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00a338a92e4e7bf543340dc849230e75.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b2e635ebad8aba82656c8fceb05018c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d3b46bdc45c8bc24ed89cff69f06ab3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d103ca7cf575757374f8f6ae87d8868.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f23ce5c1129ad70c079d262448d9fac.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10f790eb0fe5653c1980ddd983ec79c4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16acaa39e4fe4edbc2becbeb85e8ccbe.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0815082cf02417e80f4df9dfba17aa13.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0159b6457aac89f43d1c1931cdf7500b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c2fdeb77e6e650e69ecca013c7e8e67.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c71338af3f3a8c068ec28b5f6b8f8ae.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/169db2d825a1bdeb5a576ab746812936.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/136b8208ffd0262ea1aa6f8c17265ff7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06412667e714332186d0cf86375e98fb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/141706c74486585c27de2dfb335695bb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11730ea41395edaaa3a80d757c7dd1c2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10b7de79c1ce32cb8928616b00c3686e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/146be641443a270dd8116f65d53d0c9d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c9bac77cd53f1c6756c7581e9b0bd00.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a9f8c6ceadfff61072a565311777655.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/007ff9a78eba2aebb558afea3a51c469.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1195f58740671fa1d73c91f39031e417.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b9be33db71b9237002df13d40b7282a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0df400016a7e7ab4abff824bf2743f02.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a688b0783f472c84bbfe1c56efea1c6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08b49c79fafad70a23c770b1fcaa89ba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01f8540fb1084107a6eb3e528f82c1aa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0177a92a906192bfde8adbb8a237e524.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f9d92617b85252200cf99ef0f84d59d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0672592809da7574a87ddcd0ba2f315d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06d80d6a7aba8bc48606a285bbad0697.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/088cdda83ef0920d130398f7724b391f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08782eb5ac167ca5f8feaccb519e9b4e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0acfee91dd38e53a20dd1488a17e9af9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06bd6a364e68b886c085b9b8b8d2b818.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00693b8bc2470375cc744a6391d397ec.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0440e760e55c4d6bed536fbc0273801a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08e4013c0ddc710c57d1d188bf7c0cfe.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/156e74a393c76b215752d692eccd9135.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f0effd9b5d25fdab84c2a8ae17d6deb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05134ea3341a1f4f460168e68ec5765d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e167bb5c5d4736190d220d081d1de23.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17e00d79ad69729522d8705e95939f01.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fe613c90931a021ad3716dcb9a5f270.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d695ef1bfe438be0cedcb8ef146afad.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e63b24bfd67158963830f64b5126aef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f8e8464cfa3bacc074d6e0289d6657f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04254919cc15c6867cccfd738a926999.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/103e4b4c0ebb0e8dabe6d47970fb546b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/106d7e0df869e7ba9f6d16c5f77ddfb8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e708dfe2a8c60849a4b625eda57d118.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/153bbd70025d2738418176e6ab7fedd0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09839ef1c5a5a5b3acb61c4093cab07f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03ddfa6f292b49e14ed6be5c58246701.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05beb3230462b740e5c56230eb27a7a4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f91ba06bfea7401f59c6d796717029d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1586326d52312228c20c3599feb72d7a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/104f3118ce7eba3e48138ccaa201f25a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/091306dd0d110a677fe0de1ad066f160.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09d7cc03f9e9730bda53b4a590351838.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01d0f3692e576b3cde511285352aed4a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02b1c50fb7315423a664f3ce68c94e30.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0eac90a6621ecbf97adfec04596b77a3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/179bdadf902bf8447e9eebbe63553d24.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0974b614b86b829821067bd0b1808b8c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02777bdfcf9f1a593af768e6616df4b3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02c48781eff39a66ae6dd8626e35809e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a0c223352985ec154fd604d7ddceabd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0cbdda18846cd014dcb0e18fa67f3f98.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ad3d66ba7b2d5d6d608da088abca7c0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09d462e2ad4d8c51d1a8430577891d8a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/023e4e28415506e0deddcbd8f8bdab29.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16d21426a864e73050afda40bf1fddc6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0229677aec75183a16de9d6f6658fae2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0811e959c0fa3b1c7af6be4f645a9464.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1853938d8adf8942d927f5fae8b9ec0c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16a2425b83d397180e11ec5cb2c4b44f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16da5fc3f2c4273902da11aafc13456e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ec6e2d3078a83ccc3f7fd5a4ac01e3e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e94678bb5498dc6df640553bf3a1b2f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ef80d6c4d739faaae8d3cc08c696c54.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/024ecfa590271db8616bfaa59159d7b2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0067dc3eab0b3c3ef0439477624d85d6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/098dc5e173d21c229e6ee565d7666b10.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d2129d0eda43dd8514ea309c29ddbb0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0120343862761d052d6a7ade81625c94.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/011e06760794850c40f23d6426c40774.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07eeb2ef42c6b260684e8cb8e8c4ba70.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03edf850d74d43b2587991ecb673fb33.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bf89dcb09d8c76be568f17811664560.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1548eb783093ac6bdf011d07f3370a36.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ea78b024dc3955332d3ddb08b8e50f0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12c65c69fa9e112448f7d26ed34abb57.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/025d903b457b865fd29424916e42fbc2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a438069d5a8a8a079ac131d74c3a366.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/042e83e82315bada78f2681d030b0d28.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a9322a30aff755dac328022266e3740.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a077ea0c8fa54d95f75e690b2c8196b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15305c67da0838b92b90c66526cef5aa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1574f04538521f284218dfbf2c8678ee.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/109bfcf2de8f0b7eb5f7768a51ebd565.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0428e1ae313156ec06dc42096cf1372c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/098c722479a8288c57b545781d5c54bd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/165f26ae3d9dfe88e10f7929c031b1c8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/139fc3326016bf24cc06ad898c42742e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17738f5f90c73ed152991b28ea4d77ed.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10c2616bf8e750b282cea404d01a3429.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0944fbca92fad9a38af10b3d1943cb54.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/174c68c77ca368337e9fc35112d7b842.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12b72be0bcd7a96172fee4f7cd5b01fd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/060112a1a77217039de21f7d0963929d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d37ea8a209eb60e9c03eaedf3fd5384.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/080ec59213ed7425b7f8c169955bb9d0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00792e341f3c6eb33663e415d0715370.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07d7cfbdbd3682cae50902b53c798028.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12d1b8763a1b74c7e40c65b005f91ea9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0512c1a59449a9c8c83c95722028ac90.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/042ecd9a978c2ee48d17f7f781621ac9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/182e24a6ba14bfc0f109687b22589f57.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/009509be3ca7cce0ff9e37c8b09b1125.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/087cd02ca089fc0f30841a6a89e3a619.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1041e0480ed0b88c41169fb2b119fcf7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14f4938d64e0da5582a4ced35cf33a6b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/081f6b553afb94d1f192f08cc3ac5762.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a65ba3ab9b29c66e15cec76f34eca6f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d8858e722c5f9834ddbdcecbf3cd4d6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/146c249bf2d60bf30c309341020bf2ee.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04c3bdfec0c7f082c7e697c26be9e020.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f7c32eb17dce55860772c983f557e3c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0246f44bb123ce3f91c939861eb97fb7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dc45e3e57bbcfccc550479d57b39951.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01b09bdac592b0eb9909dd105314ad3a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f2fbcae9ec83165d1ce5f5783fcb903.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06d2d84b936225e6c853a0453cb9c878.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14a2795a5c3d69a5c18a63aad8718003.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a6c192b96e55e2ca37318919b1ffae6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03b9ab173e5862580909fee10f0ea46d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01de1f7b7b4465a6d130a56746af66a6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12078d0ea135483e4bc48e2f13678588.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a9832b18d360f50f5b3b2ab4c540ddc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0457342c36e7e7d103fa4b286a1d62f8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06557d54077c5265ecdcd7273c9c38ca.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0581b47e4c3890fb74a5485aa4d008f0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ba12512fa4766cc4f8c288cb3b9b95f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02d45a238621c3f2cbde5c1d173ead1e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/170dba137a3990b225bdf19074b97023.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c4ac597d196aaf07f3af039b8fd6925.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/143ce5343024462044b72de531b5ff08.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e8d2f2fcb0efac6b731c191beb66559.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02f6152a9401568d695234d33bb6c37d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02cc0d7c8b7b513b4325bda30dce222d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b5604e1cdfcb595ceb36424126a3e09.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10f7757fdc673e159e47ea20834ba551.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0162b3e0144fb1d1ab82fbff3ace9938.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15137b6e02d5cd04cabb34aba1fabb9f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/166d9a55197d0c21068273cca67fe2aa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08e5e75d29184a82a9dada752c9c4afb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06538c8ad646eb2ebbbbdda1c8174899.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/013aed490e0e15a8989e12ae0d0ccfaa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/085faff5f472098801b371f077b2dc8b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0486732311e9d60e8712f1dc33c4ebe0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/18aa1cff5dc0615c75dec0fcc135be3c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/146c8ed2e65ed9e04d8302bf1d96815a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/123ffeb6b374097856bffdd11b693668.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06223770cd0968a5fa81b2898e65e34a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a8b6985bbe58a58909b5ef0a7d5a1aa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10730dfd280bb4b2723dc677378f2ff1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0755a940eac9a9b8cf0328b4be062096.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0822709f6f6ea0be1928f52b8eecbc2f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17c25d583276276876ecf58c011aff88.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/039e61713398f27d027480e4bd0056d1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02f07e7e51662336a9d8c775a4eac5f4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1383bc8b2db4943fafe13b7f289f4a03.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/014c2b0cd8e3b517e649cecf8543b8fe.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08774c223487c1c880b447aae6a7e0b2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16b17dfce7906f0e7a599a538c7ac106.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/035fe39454c2ddb2f6a37146cacf0ac0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b346cec75418439f13eabdc7e96f33a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0995f642fe4baab3cb3534725dfedb75.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f46feccc1e3e8729fb680a72debb29c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07bbf36ce2c7407751219804b3d187ec.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/008887054b18ba3c7601792b6a453cc3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13fc4571a3c549715a357d22fd16eabb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1399f1e08abd77f1b736303d4101d51c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13f939b9b15ec56409a21017263db93c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09c62a1ff26b3e83a2d476c2add054b1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d30e235aebdde78be6e2a59becb582d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f2a90c737499a7d8bf26bb5348874da.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/010d65bd29d246aea53d9849da142ccf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/041d0d6a8d110b35a3795dd5c68f9a36.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05a747c5853defa2420b976a9c0918a2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/145ec132c2a1cc4291bd774c7f78c871.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1043003fb33c5fa1d094cde3c0dd1fd5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12bd4b1926c160fcd73ae48215e2b12d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14e8bc7e0eaf52be607e0a654922a826.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03bc1c631e57ba87c85b98efd0912c00.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06faa041b335551e3ccf3c239d006425.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0eb297c3d5f6cc93371f02f4ed35879c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0df4124761f7303a0080d50377f2ec7b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1268223049f00d74f20d4f2af3126234.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/007b8a07882822475a4ce6581e70b1f8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02ff77af410e966b7b661f6f0789d947.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07d1dd1576b5e95f448c1213b6812c40.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05f59d40acf74affda3c8940ba192f42.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/100965db7a00c9752519e342ce9baf99.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17a79093473687e2cb3d0cc29c24f3e8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/045ea2f41120606e2c5ae1315cebfb46.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02c90d8109d9a48739b9887349d92b1f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f0f297dca75b5c780316a0f4169a950.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ec8020bc2c4bc1646a81737be8580e0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0434ddceef3eea9c757c8e9557a2d698.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/073f0821a0842917e6e97ab322bd63f2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12db63491c6590c8ef38dc1824aa4b81.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/118060ca946af5cfd1b3bbe030f9003c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11d669406a273858a123d1d1c87354c9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10c9288e30af850676d34c18fb7c1632.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/181e91cb6caf6739478d06231faa053d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c8fe33bd89646b678f6b2891df8a1c6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1831f3ce615ffe27a78c5baa362ac677.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/012ff2d21dad14452ea16b4cda7eef4c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/164a194e9bf8819523235879eb0c2698.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03c81a2e78dc915bc515fcb8aafd2f6f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14c00ab356d9261ea220ea91fa20a42d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03f6435dc8a5e760d19e67831c6034fb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/007b5a16db9d9ff9d7ad39982703e429.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c3b282ecbed1ca9eb17de4cb1b6e326.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/013f8fdf6d638c7bb042f5f17e8a9fdc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17e3b8acf23f7943b04dc680fc1bda0d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16fdb4a145a7f5695a264ba980e23bc3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c71084d955bfca989d865cbcf8cf8ae.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/173493daaae8804e5a837ddfe2e21c50.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b72856996ad5826afff7195ff678589.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06ce71ab05143b1fe45ba8eeba2e97e4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11a7275abdc2e78ff72c59e59189d1eb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/155052199dfac47b3a1ccbea97ce648c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/052675b01ae0a4b62e090923d131f4ff.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00a862390341c5be090dd72bd2bc19ef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/173d5eb6a645eee683442be8101d398e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a77d498ff491945347bb895d8ae4008.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1732acba109e3b1cf23efca0381c9a32.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/089bce55f74e34940b3782e11c1bd2b5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/055429c6fb404af27ac52a08216e6cda.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0100f55e4f0fe28f2c0465d3fc4b9897.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17b6be0ff31a2b1e417f6a03cd3ba32e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09037f0995111a413fd8e976f3680f09.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b345d4f2434903c374ad8b8513a289b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17c3f951feca0716c023ea462428dc14.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14babec7fedcfcfe575947797e2e01bd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/114a27c91c2ecc5818688abd63461c94.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14918e76fc079738923ea5ab8d12b4ee.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0484a6cc686a07e1edacd0fb44b1d965.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14cd98829e4e99e950af3cbf94ad1734.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/151474b992b295f69547839e31271dd0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c1d98a09381f4dfffc510fab188f189.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17d2ef6638b31a9be935774b2b873499.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1465759436b4acd2748dd50108b90ca9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e47c35f24067b19557e15fcfd48778c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17c5e8815cf0d086090a07b003b9b036.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03dd99e068f1f2283011cc305f5aaa9f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/164533d27e5827842f28f100196bfe7d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03b7320cf8054e57b499e01d1cde0644.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13d470408dbbdc20b38ea32d3752edbb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/187a741ca8f22a04de67b60beb12987c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12c222ec2e7808183056af545a4bc046.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02a8ed20109bd62bd5894f276c08c8a2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/053106ed091eac5e129225315ef6cfcb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10d766340a5c0038f2b0529a63ecf2ba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0270276158566af50ef48b1284c998d6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a1f8334a9f583cac009dc033c681e47.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a13f9596e03a9f87c96aaff6e057048.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12b5c6e825ea11850e23f233a54981f3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e756365807ec27c6dfd944ba5b442b3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1369c16ecf370cee5e4a306eebe5656e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/106a740e7e34d80e35b0f40a3d96fc1b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02df8523833c432f90071f86dff69a8f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/015b4aa50dd9b4a0d26dc0c38f41b489.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03b0e7ace52c10dc4878f60307cdebe3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09a1ea62ffa4b9389c03162bb0b0b572.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1716fe350f1339e19906eb2889960c9e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04d4918090b0c2a7b965ff58f13b2ecf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/100f347ec78a42a9e7c2418e4beb3f6a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14a24af1ce8c796dc96bb45dea1fb8dc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12e15979a85e881613f8f2cf49de08be.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/003df8b8a8b05244b1d920bb6cf451f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13f765ea64c7a3575a105cbdfbb6c31f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00fda6ecca54efbac26e907be4b0b78b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ab9cdefc666573f8019356ffefe0c69.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fded83899cc06dc30c1bda6302b5bd6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05e3dfffd0b0dca56e6b1d4686e1a6c2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07466e87275ed10056a9d2cbb4c733d8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/129cff4506469af7863a6e30c9947a66.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e0e44349e6955bea9fea91de26a6980.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f3416f29133c2df1b00be4448dc4473.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/120e665f7fa566a429d27ad920fc34da.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15aee288781e5c11d1f9a6983326ad9c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/062131c6c38a1e485091c8c3329d9638.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ef688efcc98bc287976c3d4d8145ae8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/072aa9d6100187288ef00316c8bcdd66.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09fc0a4acd13c78c02cdb251f193231d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03ae71ff9e4c5ac97afdfa956ea5e191.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1392df2ceab56cdf380813987c7c9ba5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fa361cea0655e945970cd4762355ac0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15a4ad475f9c8b2250df082b8e741d94.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d83fbe9ab3684226edd81a13411bdf8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1596cf0ab964b3ff0587f0d30042ca18.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bac7ab4e3b6c7a5331067827cda04e6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/037ac6cbd9c96d70e68917aa059b6aab.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0379145880ad3978f9b80f0dc2c03fba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11bbd6229f14cc68bd0f157a9ef1d47d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16d2d5edf52892450e38aa4f766c2eea.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1242d1521d58b7cc6dd03d95a02c8bcb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1012fef5c11e2875d2268c93253108ff.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/085b26edb3ee9688d41c4293aafe0162.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b1b49b8cf679eb3e2fb13ca6deba9b5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07ddc3c2188164b1e72ae6615a24419a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/044f7e60953230fa45e4367073d96693.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02ef5f19db8cf4da62016a1067bdc548.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14cbd6c6c6550fa1306c8d4f7cf1a840.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0eeb17d40281f3bd639de24ca662c5d2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0360d42966d191a5db4c4bb2e8c66229.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/176e2a6cde976022e90b2b42a81297fb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0496d4170e295688e8d6929b239dd4fb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ef07f4a6706a04af9ff354e263a28b3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06c47d61293f2bb51a94040bb4b20799.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/071652f5f5d0bf00983c075e96ad725f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0998eda08cdb14648e444c15f06658af.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02eef0f899b81ddf27a42641bc8db9ee.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/044644ffbec4b6d402eb824532478811.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04be608c9c005d9aa0224fe08554e4be.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a783538d5f3aaf017b435ddf14cc5c2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05b5e17b96409ca6db51edaf28bd3bdc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a001d75def0b4352ebde8d07c0850ae.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16d9e14772868ce890909983e53af8d7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12fbfefaae57137c745f16b0cf11b84d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1634d504adfb4efb6b14e891e8996585.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/146fa113f3d8ad6e0effe9719e09cef1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/159d07f089673cd225f3a47d08ee772f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0311a6a51a414ba91f3ad8055170baa1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f9a30195723e9951f65ac1246e5deac.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/099db37801090ed56bbc95c13d3799b9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1381239a87d5a97caab8d0cf72a06e75.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1312d2ca6f4bfd594624f7161be73ee7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/123c19c8d168e7704273cb7174351821.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/166da5b561393f0dda932f562c39bc6b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04fb4d719e9fe2b6ffe32d9ae7be8a22.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fa7d7d2e948948c67919565a380ad45.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/168e2da635938b82819b8a45bbd3dd0d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12b839068a6c29541797fa9bd20f350c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13f6cb46cb23d6eb41fa6931b73b3e4c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1699a179453b177a5e895c5b5ecb3624.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f378abee01a23bc9e651753300a9a36.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/153f05ef64f050530ea746a357855b20.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12de62fb1fa5a48d596428dd5a90184a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0959e919604a931fc1ae379b7a75911a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17e74cb24060ad496aed8ca01f611ba9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c36c19e7c4e932b8e0c01aa845b2fce.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/065221667b1ed903e364b9063ee2195b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1428026a4472cda58ad13233bb9ad64d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a86c8b48bddb0155645f3a27f48292f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bf36935030d2b842b86526bdb367ff1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02ad779f2721db9364d3bb68f5580582.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10edc6f98c2423e4e195e09a86b0d292.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0569f1d79637070f70ca4a62ec510792.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/152c0d164f42991d55d84dc30a1fa81f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ec9be8b32f2b9eff2b817a7f722b118.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1069587e556525ecf7b3c6def3c096d7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08ed3bf636ad906f5ddea9ebda22ff98.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/182c436dd1fbec1c5f4979077ce74659.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1493bf7f5cb1ca62b4e0772d9edc702e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/056b535b441278e83839984f1b1da0a6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1381df385b4c5da43539ab48633bd9b5.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1081c1b755f05b219275afd0989c6748.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07a4fcb31d0c6259f5ec21a1f193bc39.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fc6556b7504fc473d8bdaa0a555a4de.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09d497888fd1e6745f67a78269139620.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15f8b879f8660fece35ed760345fa9dd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11722136645ee54db58a0df76eae2179.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0883e6cc994d9ba90592f400d502d838.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/060b163b5ad74c889c47b9f421825cdb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1278143f78dc063c1e29970bf3eb32a2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f6d2d6c124fb61ba7c59ef8dbd42c99.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f7db98cbd6f6537c0ac3dfad349b182.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06cd8e3fbfe72c34461814239e1deab8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/092220a7a8081144a7485efb6c087fe2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09477e5059ebb15cad5f37fb3b9b1889.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0518691772e78ac6805bf006993665a4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/173b54d60d2d75be13416af93098445c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14158fdadb6a3accea6b334d0f284092.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b465156a0844773fd47ce40d0dcc7db.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0857307dddb1b41288887ef2338af6df.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c24a6dac8f0dc55edeb80a7b683af5e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fa302727af44e7ad8033825cea98d7c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1852241f20c06eb5d999cd43cc92bbeb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16ca736a6ff68bc4d2d1586e8eec9b28.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/162a3de12cc110679a921dc49dd38fc8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f26d3f7ff65d0713f36cd8cfe6c7a5f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14cffe576239620f1733dfe487dbaa6f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13e608df9c88b1465ebf98bc84d0a832.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fc12a365adfcbb603e298b10149632a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14acf981af889d9b97d1d46a439228b0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dbf0bd0fdc7b594de02e573b1a04a56.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1758bbaf1dba23fd418911be2a2becc2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e3cdff3560de43a8aa1d9820c211fae.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13b5f36cb39cb5123958c6e4fbfd5399.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0207803a6ce1bfad98e7f095c965e44a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/059a86d50634c78d5a18f918cb0b3f0d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1132baf0ad2beef88c9d78c8ee21778c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03cae0a6700c5cc4900e576034361e7a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0cbca2fcad7910a3c98d734e17cfde65.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00fa641312604199831755f96109fde7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d4e09456d9ebfdb076e34326586f18e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03f5b638b8f1c83d3097786e40992ff9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fee7eb61d52b589414845803d73f2f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15ad574a13084df42ef17a5f635fae3c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14dee67f611c52183fda5fc07db5d5fb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14b1b3f0d45a15766dc02dc84899b07a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1710f33e5a5a32fd20547e8f133fa8d9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/122abc906f3215996cbcccf63c8fc89c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/024fc0be976885d5cfcf6770239a9001.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ddbc4702eff570783cd03645571e7f7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0593e37870ee77b0d34508e118bf6670.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12590e7d21b4e221217c7fc8fa67a800.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e1fcfa1796f8c54b9fa4b56a3a1fc4e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10ebf7599dd41ded00fd74bdfeada500.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0afc7d87053cb44a529c78c0fac99886.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d96ca29ff2e557e93c3de0ba017ff61.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/159d8e089f2ce2a38580a02318fd9ecb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04d3a777baa532f7558a860393c4537f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/098333d51970304787c7061f3e683c5b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13b1ca09a561661a8c1a506d49b8dd83.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0990fe6be15d9d556eac8712db3c8094.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12fe7ed6bd250509b131d245551b06f6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14b27fd74697cf699143ade2d4f801fc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04f9e2da4b6f5adb9b9d62eba66ba991.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0021f9ceb3235effd7fcde7f7538ed62.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1304760775c1aff0f21c311bd48f9580.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fb1454104deee316f52244bb2037b37.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16cc38641158115b57a4610e40e5f6a9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f015731185a2308c1a85eed4e8ad728.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1241cd8dc2ee02844a420b6d0afbc97e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0d6bab298a320221f08c49c701d3e06f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11de51e128d743d0d40dd8ce3b4fdb92.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10c132cf556fc31a60c0f07e0d416a6c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0255a5bca3b9d91d4fdba8d7419b5e69.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13ea49cd2b706ec9a15e7214e492bbb2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11e572c36d23d362ff987d7f8828dd45.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e1ac042292a573ecd0ec71b3b646997.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09beb7445a14486752bd25f69e952ebc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c1a1b2adfd0aa2fef3eb974b9b95734.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/013c43f40f0fb13fa25ac0c2a70fd48f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00ddcec076073cc96f82c27bf4548fbc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/135587061f51e7dabb820bbbd619977a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1476da6f7fc9421c8ec3c0aba6b3e1bf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04a3f5dc6d985601f354ab4434645d83.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0326344e0d5181130c28d25edd5627e4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00b7d114bc5166a629a3cc03d9329120.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/156d6d6ef7da47aa22b150268af9a3c4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/055cfac66e5d67756ab05d02baeeb04c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bad79c06832a1bdeaedbbb50a28450a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/131ccf48aba2dc53174c170b2429e55b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/092c2cdbf133b201d7d56969bfbcc8f6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06fe22a92fa8dc8f90e7a0f118acad6f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a135c5ee55edf4bd9631d4e15264788.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10706a4ccdd719191686b98d4dc48ed0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/097ef8eca8fc340249b0bbc2b745972c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0e934073a6b9505c0ded4e34437ca538.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c0955cddf9488d30e6e8d4ee7c18856.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0345638400c47bbd920474672562077e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/018aae5768833c34e3df30d63429251e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b3d3120a62c2bc280f589d473041d06.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fc4930f411a11df34283f349d891654.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0310a425f26d797634aadeb66bd63ef1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1804c9f9d48a34ab283e3a39006936f2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ca382d7309e346ac802a202d0fa8c67.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1472fb85b5c76d50009137da3dbde052.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/17a2b04e47d6873a13d999bd605d0427.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/019c92d1ff56deb9d75d809a92bc84d8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1411967eaa020b15874e03167eceea62.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0075dc49dab4024d12fafe67074d8a81.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03c1557464297ffffc8da977c4db7e68.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/011761e3ae8ec5e188e8f0bcb59bb8b6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0be68a5e7b40e9689c3ef2a9764bfcf6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/093a146e173a3fd9ea650b87f08d6222.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0450c3899b7ca7cac31406a52c7bc74e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12eb341a614941c33777c708d43a042f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04f69d645182d799dbb54862c19c0f38.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/139b8328a97a600cceb73b0875f80a98.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/015b363b062f602e7ec04ce28e640d05.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1553cf296f4d83f015a07afde78fd747.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08149eb72785553570f74a6effea8d7e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12a96a9c366b60dcc62fec29006780a3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11b60d8d86f14a601ca290909a17cbc6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11da2a7ad0326cbf4b46da32b1b82bc6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07192213791150248bfb5bbe6b0b0373.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1143bb719660680ac0174ce68dc16773.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05d6e6de49a6999e56de23c4608af441.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11e29ec9659995ebdc80f8d812dedaa8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b102a6aa442aed98f137616e924e18e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ac12f840df2b15d46622e244501a88c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1880c57a7ac87a232158207581c989df.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1416047ba98d2c23eb2caf45d3008ccb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/150a95102a32d2f099b02a445442310b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0653d31ae648e2a6628ee3f440729361.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04b5834c3a969c6308ce157c547bb313.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03dc61595ad9dbf49e3998cf586ca8cb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0db44ddb42bf1f97de987abe2bf01839.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/11f968569f31c60b73a6d50b907f800f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/099b068027c41251d29f75a311cc5e5c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/114524d11fe49af5be43897233a4f65e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b673cd41260b9fb998b0a8ad164b6d6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1435300899e9180c90de6e100643b402.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/131c49315a5bc73880b305a844bf531a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0287b3374c33346e2b41f73af3a36261.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ddf7ab49836b36081621a655f3a5c21.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0964e161b379740955fa95d8d4c8697e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/001513dfcb2ffafc82cccf4d8bbaba97.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/16ed388064b485a047d02a147e0a5cff.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/182c9134e07c883dfdc2acfb21184810.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dd1914e122292dfbb25e35537b511f6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01ee3c7ff9bcaba9874183135877670e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04d2dc1f2804c748b1aa71954bb45d38.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/00f34ac0a16ef43e6fd1de49a26081ce.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/132538caad57960ca3c9162e2f4a8498.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c329020d6ca0ae5af30f6857ac7e86c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a4f1e17d720cdff35814651402b7cf4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0ac12e1849fd51b2158d16b8f5e75551.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08af52c888b0f5735c3ced810771601d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0234695e7a548e2ad1b2ac91d6486c1c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/081ee1793b0e0027f8cccf9dc4513d53.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/15e9e5a6a6f3630335261259940114b4.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dde7ba887ee083616993d5892db139f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/084f941981c326f26ea77158fb449c57.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13267e83358116ad51bebdb1be3d0a8e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/059513b85ae976a6f591f71bbaae49bf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/006cc3ddb9dc1bd827479569fcdc52dc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07a0214fa84969b5256bf7d20f1b3a9b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0eafdcc7628b2c74983819f40c763c2f.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/095d868f796f86c8258750365a04f2cd.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0672befac9ea0c8fe82c37b5a634d87c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03eb844e1d5aa5addaab20298b1b70c3.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14914d7f0348c56b32a779596c11ecd6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1866bec18870abddd279fa4b1f0e126b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0fc1bd3c4d89b3f89dc91605032fa8f9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1418b4cd78095ff2635ef02684a026b7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b9d58d8d2626f80bbc5cc3ccf7e8bad.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/181fadf5f2222c0791c7a02fad19bab1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c207c7af2d1ef71dc61434f04d00cdf.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/146b732430bb63d5338fc6b7a8fc0135.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05a05cf4c8d6a4f3f780a9112a11999b.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0df07a23ab97135ce7d390f2d12e388d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/153f0c739e9049612a62f111b9519429.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dd7b725203b91c30ca4f796d168e6ba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10c9ac7d54eaf2a4ed12883125afa4e2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/045d421a681a7da04ff668992a8b4c3e.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/136e0fc1f67b72037c0e60faff6061f2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/113c17f9199d16d1a9a1b16e3740ef78.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b6239db9b1649fe2f513357c82931aa.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/10a59be0ffb5e269849d409d05b1e94c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0c378307f600ba007a7eaa50fba6d0b8.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01680178ca6664294efb493a46014a82.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b57520c27ed32bc21e35c38cb5dd268.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0dcb8150908bbb7b7fff1b9d41a358ba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14d2892358ea21837f7a9399c0b00acb.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/12a2e809d7f15298050f3798c018c395.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/05e25c8ec67962c2ef4dd7b0f6aa3918.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/008ba178d6dfc1a583617470d19c1673.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/079e17a742380d333950962f2c83ffd2.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0afb0d170c66aed12805f838041411ce.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/03da788cc7e5c111575daf04e7e3910c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/06ab2182c2a958a5f0de2035f39a0fa7.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/174878bb8d33d52381bb3eb36dd1974a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/046b28d8ac6f7beb3f63159ecaf4cbb6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/094b97e52d00895fbf6769724e1b0e7d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/13ee74fa50b3e32356047fde4fd993ab.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/09ea184490e5da33eb3d851ee6361941.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0bbe00ace11a4dd944b8bcf8ca9772c1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/083820e6dda52da4c0f200ac36f582dc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1312009d41a9488bf5dc0af0289b0657.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0773acd6874a703b966367d4d27cac71.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/101c7d1a68a7282bbc5a1a226d81cd74.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0314048edbd08c8c50eb3c93281bd043.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/102b81ed975bfa49e496526720e9b671.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/1534c4b9f44f10bfb3e6e0ff32d08a9a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/025bab46ddcde249d5c52c660fea6d26.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/08ea655aa168d87806340c336d07f1c9.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0214e2d616ec7eb7cb0d8c19a7f29e70.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0adfec2804a1efe41f54758cd87cec01.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/07f1b6214597af2f04a27c375a602a0c.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/04ea8788cfa17d01c87ea4cdaa0a330d.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0b2d69f59303805f17055bd59615a297.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/02e7b5a86cec57f5935374946d4f5dc1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0f04466edd10d6c1d27e123399cf4433.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/008b1271ed1addaccf93783b39deab45.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/113ef2bb9e14d89f927314f73d573313.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/01b6c2bd3053535a58d8de763cf06aa0.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/029202b0cf0b5d6d48c6ce7b432409ef.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/065452ccfffdc6b6f60183bd6c88ba89.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/14f25ad78f02126a1ceeb44385378cd1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0a70f64352edfef4c82c22015f0e3a20.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/train/0267c5f1acbab52ae4a7927e0398612b.jpg']




```python
len(X_train)
```




    800




```python
len(X)
```




    10222




```python
len(X_val)
```




    200




```python
# create a data batch with the full dataset
full_data=create_data_batches(X,y)
```

    Creating training data batches...



```python
full_data
```




    <BatchDataset shapes: ((None, 224, 224, 3), (None, 120)), types: (tf.float32, tf.bool)>




```python
# create a model for full data
full_model=create_model()
```

    Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4



```python
# create full model callbacks
full_model_tensorboard=create_tensorboard_callback()
# no validation set when training on all the data, so we can't monitor validation accuracy
full_model_early_stopping=tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                                           patience=3)
```

**Note:** 
Running the cell below will take a little while...(especially the first epoch)


```python
# fit the full model to the full data
full_model.fit(x=full_data,
               epochs=NUM_EPOCHS,
               callbacks=[full_model_tensorboard,full_model_early_stopping])
```

    Epoch 1/100
    320/320 [==============================] - 2084s 6s/step - loss: 2.4445 - accuracy: 0.4740
    Epoch 2/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.3876 - accuracy: 0.8874
    Epoch 3/100
    320/320 [==============================] - 40s 124ms/step - loss: 0.2283 - accuracy: 0.9392
    Epoch 4/100
    320/320 [==============================] - 40s 126ms/step - loss: 0.1409 - accuracy: 0.9682
    Epoch 5/100
    320/320 [==============================] - 40s 124ms/step - loss: 0.0989 - accuracy: 0.9823
    Epoch 6/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0701 - accuracy: 0.9900
    Epoch 7/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0556 - accuracy: 0.9930
    Epoch 8/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0399 - accuracy: 0.9965
    Epoch 9/100
    320/320 [==============================] - 40s 126ms/step - loss: 0.0345 - accuracy: 0.9977
    Epoch 10/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0299 - accuracy: 0.9972
    Epoch 11/100
    320/320 [==============================] - 40s 126ms/step - loss: 0.0237 - accuracy: 0.9986
    Epoch 12/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0212 - accuracy: 0.9978
    Epoch 13/100
    320/320 [==============================] - 40s 126ms/step - loss: 0.0177 - accuracy: 0.9988
    Epoch 14/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0170 - accuracy: 0.9988
    Epoch 15/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0169 - accuracy: 0.9984
    Epoch 16/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0144 - accuracy: 0.9986
    Epoch 17/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0094 - accuracy: 0.9995
    Epoch 18/100
    320/320 [==============================] - 40s 124ms/step - loss: 0.0114 - accuracy: 0.9990
    Epoch 19/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0102 - accuracy: 0.9990
    Epoch 20/100
    320/320 [==============================] - 40s 125ms/step - loss: 0.0094 - accuracy: 0.9989





    <tensorflow.python.keras.callbacks.History at 0x7fb3c9131350>




```python
NUM_EPOCHS
```




    100




```python
save_model(full_model,suffix="full-image-set-mobilenetv2-Adam")
```

    saving model to: /content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-12431614343383-full-image-set-mobilenetv2-Adam.h5...





    '/content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-12431614343383-full-image-set-mobilenetv2-Adam.h5'




```python
loaded_full_model=load_model('/content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-12431614343383-full-image-set-mobilenetv2-Adam.h5')
```

    loading saved model from: /content/drive/MyDrive/Colab Notebooks/Dog Vision/models/20210226-12431614343383-full-image-set-mobilenetv2-Adam.h5



```python
len(X)
```




    10222




```python
%tensorboard --logdir drive/MyDrive/Colab\ Notebooks/Dog\ Vision/logs/
```


    <IPython.core.display.Javascript object>


## Making predictions on the test dataset

Since the model has been trained on images in the form of Tensor batches, to make predictions on the test data, we'll have to get it into the same format.

To make predictions on the test data, we'll:
* Get the test image filenames
* Convert the filenames into test data batches using `create_data_batches()` and setting the `test_data` parameter to `True` (since the test data doesn't have labels)
* Make a prediction array by passing the test batches to the `predict()` method called on the model


```python
# load test image filenames
test_path='/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/'
test_filenames=[test_path+fname for fname in os.listdir(test_path)]
test_filenames[:10]
```




    ['/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e334f758c7944df19c98d49498d28c64.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e19900119c7d3ce48d4035cd0211be72.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e6a5c18da7beedb1622bf7d18b452121.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/dd3c80cee38d165aaf48083f4a4a0071.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/df0d6ba158287cb2b3ed6459a22d42ba.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e37d651b9b5fdcf26ab37259fac877d1.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e0eea374c5170e90dc0b1ee795470ca6.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/de2c6675c001726a96ad9d72ab229f3a.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e646ac89e0832502f9a726c72773cfcc.jpg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e079440ee0061b92ec22faf17be13908.jpg']




```python
len(test_filenames)
```




    10357




```python
# create test data batch
test_data=create_data_batches(test_filenames,test_data=True)
```

    Creating test data batches...



```python
test_data
```




    <BatchDataset shapes: (None, 224, 224, 3), types: tf.float32>



**Note:** Calling `predict()` on the full model and passing it the test data batch will take a long time to run.


```python
# Make predictions on test data batch using the loaded full model
test_predictions=loaded_full_model.predict(test_data,
                                           verbose=1)
```

    324/324 [==============================] - 2251s 7s/step



```python
test_predictions[:10]
```




    array([[2.87460239e-06, 2.95268787e-09, 9.77277104e-09, ...,
            1.51160451e-09, 1.36108938e-04, 1.30030870e-07],
           [1.72946360e-13, 2.61498006e-12, 2.52950140e-13, ...,
            4.52893140e-10, 2.12585904e-11, 3.48568532e-13],
           [1.97526648e-10, 1.88702484e-10, 1.68101592e-11, ...,
            2.61903499e-10, 3.44374984e-12, 2.82499668e-09],
           ...,
           [1.14090428e-04, 2.24341448e-10, 4.75618822e-10, ...,
            1.51802293e-09, 1.23144000e-05, 1.01480397e-08],
           [2.47080861e-05, 9.72131602e-07, 1.40727508e-07, ...,
            1.16987085e-05, 1.21413896e-05, 1.25572751e-05],
           [1.51851483e-11, 8.55570170e-10, 1.42025139e-12, ...,
            3.78402202e-16, 7.13462086e-14, 7.65905826e-15]], dtype=float32)




```python
# save predictions (numpy array) to csv file 
np.savetxt('/content/drive/MyDrive/Colab Notebooks/Dog Vision/preds_array.csv',test_predictions,delimiter=',')
```


```python
# load saved predictions 
test_predictions=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/Dog Vision/preds_array.csv',delimiter=',')
```


```python
test_predictions[:10]
```




    array([[2.87460239e-06, 2.95268787e-09, 9.77277104e-09, ...,
            1.51160451e-09, 1.36108938e-04, 1.30030870e-07],
           [1.72946360e-13, 2.61498006e-12, 2.52950140e-13, ...,
            4.52893140e-10, 2.12585904e-11, 3.48568532e-13],
           [1.97526648e-10, 1.88702484e-10, 1.68101592e-11, ...,
            2.61903499e-10, 3.44374984e-12, 2.82499668e-09],
           ...,
           [1.14090428e-04, 2.24341448e-10, 4.75618822e-10, ...,
            1.51802293e-09, 1.23144000e-05, 1.01480397e-08],
           [2.47080861e-05, 9.72131602e-07, 1.40727508e-07, ...,
            1.16987085e-05, 1.21413896e-05, 1.25572751e-05],
           [1.51851483e-11, 8.55570170e-10, 1.42025139e-12, ...,
            3.78402202e-16, 7.13462086e-14, 7.65905826e-15]])




```python
test_predictions.shape
```




    (10357, 120)



## Preparing test dataset predictions for Kaggle

https://www.kaggle.com/c/dog-breed-identification/overview/evaluation

Prepare outputs in the required format:
* Create a pd DataFrame with an ID column as well as a column for each dog breed.
* Add data to the ID column by extracting the test image ID from filepaths.
* Add data (the prediction probabilities) to each of the dog breed columns.
* Export the DataFrame as csv.


```python
['id']+list(unique_breeds)
```




    ['id',
     'affenpinscher',
     'afghan_hound',
     'african_hunting_dog',
     'airedale',
     'american_staffordshire_terrier',
     'appenzeller',
     'australian_terrier',
     'basenji',
     'basset',
     'beagle',
     'bedlington_terrier',
     'bernese_mountain_dog',
     'black-and-tan_coonhound',
     'blenheim_spaniel',
     'bloodhound',
     'bluetick',
     'border_collie',
     'border_terrier',
     'borzoi',
     'boston_bull',
     'bouvier_des_flandres',
     'boxer',
     'brabancon_griffon',
     'briard',
     'brittany_spaniel',
     'bull_mastiff',
     'cairn',
     'cardigan',
     'chesapeake_bay_retriever',
     'chihuahua',
     'chow',
     'clumber',
     'cocker_spaniel',
     'collie',
     'curly-coated_retriever',
     'dandie_dinmont',
     'dhole',
     'dingo',
     'doberman',
     'english_foxhound',
     'english_setter',
     'english_springer',
     'entlebucher',
     'eskimo_dog',
     'flat-coated_retriever',
     'french_bulldog',
     'german_shepherd',
     'german_short-haired_pointer',
     'giant_schnauzer',
     'golden_retriever',
     'gordon_setter',
     'great_dane',
     'great_pyrenees',
     'greater_swiss_mountain_dog',
     'groenendael',
     'ibizan_hound',
     'irish_setter',
     'irish_terrier',
     'irish_water_spaniel',
     'irish_wolfhound',
     'italian_greyhound',
     'japanese_spaniel',
     'keeshond',
     'kelpie',
     'kerry_blue_terrier',
     'komondor',
     'kuvasz',
     'labrador_retriever',
     'lakeland_terrier',
     'leonberg',
     'lhasa',
     'malamute',
     'malinois',
     'maltese_dog',
     'mexican_hairless',
     'miniature_pinscher',
     'miniature_poodle',
     'miniature_schnauzer',
     'newfoundland',
     'norfolk_terrier',
     'norwegian_elkhound',
     'norwich_terrier',
     'old_english_sheepdog',
     'otterhound',
     'papillon',
     'pekinese',
     'pembroke',
     'pomeranian',
     'pug',
     'redbone',
     'rhodesian_ridgeback',
     'rottweiler',
     'saint_bernard',
     'saluki',
     'samoyed',
     'schipperke',
     'scotch_terrier',
     'scottish_deerhound',
     'sealyham_terrier',
     'shetland_sheepdog',
     'shih-tzu',
     'siberian_husky',
     'silky_terrier',
     'soft-coated_wheaten_terrier',
     'staffordshire_bullterrier',
     'standard_poodle',
     'standard_schnauzer',
     'sussex_spaniel',
     'tibetan_mastiff',
     'tibetan_terrier',
     'toy_poodle',
     'toy_terrier',
     'vizsla',
     'walker_hound',
     'weimaraner',
     'welsh_springer_spaniel',
     'west_highland_white_terrier',
     'whippet',
     'wire-haired_fox_terrier',
     'yorkshire_terrier']




```python
['id']+list(unique_breeds)
```




    ['id',
     'affenpinscher',
     'afghan_hound',
     'african_hunting_dog',
     'airedale',
     'american_staffordshire_terrier',
     'appenzeller',
     'australian_terrier',
     'basenji',
     'basset',
     'beagle',
     'bedlington_terrier',
     'bernese_mountain_dog',
     'black-and-tan_coonhound',
     'blenheim_spaniel',
     'bloodhound',
     'bluetick',
     'border_collie',
     'border_terrier',
     'borzoi',
     'boston_bull',
     'bouvier_des_flandres',
     'boxer',
     'brabancon_griffon',
     'briard',
     'brittany_spaniel',
     'bull_mastiff',
     'cairn',
     'cardigan',
     'chesapeake_bay_retriever',
     'chihuahua',
     'chow',
     'clumber',
     'cocker_spaniel',
     'collie',
     'curly-coated_retriever',
     'dandie_dinmont',
     'dhole',
     'dingo',
     'doberman',
     'english_foxhound',
     'english_setter',
     'english_springer',
     'entlebucher',
     'eskimo_dog',
     'flat-coated_retriever',
     'french_bulldog',
     'german_shepherd',
     'german_short-haired_pointer',
     'giant_schnauzer',
     'golden_retriever',
     'gordon_setter',
     'great_dane',
     'great_pyrenees',
     'greater_swiss_mountain_dog',
     'groenendael',
     'ibizan_hound',
     'irish_setter',
     'irish_terrier',
     'irish_water_spaniel',
     'irish_wolfhound',
     'italian_greyhound',
     'japanese_spaniel',
     'keeshond',
     'kelpie',
     'kerry_blue_terrier',
     'komondor',
     'kuvasz',
     'labrador_retriever',
     'lakeland_terrier',
     'leonberg',
     'lhasa',
     'malamute',
     'malinois',
     'maltese_dog',
     'mexican_hairless',
     'miniature_pinscher',
     'miniature_poodle',
     'miniature_schnauzer',
     'newfoundland',
     'norfolk_terrier',
     'norwegian_elkhound',
     'norwich_terrier',
     'old_english_sheepdog',
     'otterhound',
     'papillon',
     'pekinese',
     'pembroke',
     'pomeranian',
     'pug',
     'redbone',
     'rhodesian_ridgeback',
     'rottweiler',
     'saint_bernard',
     'saluki',
     'samoyed',
     'schipperke',
     'scotch_terrier',
     'scottish_deerhound',
     'sealyham_terrier',
     'shetland_sheepdog',
     'shih-tzu',
     'siberian_husky',
     'silky_terrier',
     'soft-coated_wheaten_terrier',
     'staffordshire_bullterrier',
     'standard_poodle',
     'standard_schnauzer',
     'sussex_spaniel',
     'tibetan_mastiff',
     'tibetan_terrier',
     'toy_poodle',
     'toy_terrier',
     'vizsla',
     'walker_hound',
     'weimaraner',
     'welsh_springer_spaniel',
     'west_highland_white_terrier',
     'whippet',
     'wire-haired_fox_terrier',
     'yorkshire_terrier']




```python
# create a pd df
preds_df=pd.DataFrame(columns=['id']+list(unique_breeds))
preds_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>affenpinscher</th>
      <th>afghan_hound</th>
      <th>african_hunting_dog</th>
      <th>airedale</th>
      <th>american_staffordshire_terrier</th>
      <th>appenzeller</th>
      <th>australian_terrier</th>
      <th>basenji</th>
      <th>basset</th>
      <th>beagle</th>
      <th>bedlington_terrier</th>
      <th>bernese_mountain_dog</th>
      <th>black-and-tan_coonhound</th>
      <th>blenheim_spaniel</th>
      <th>bloodhound</th>
      <th>bluetick</th>
      <th>border_collie</th>
      <th>border_terrier</th>
      <th>borzoi</th>
      <th>boston_bull</th>
      <th>bouvier_des_flandres</th>
      <th>boxer</th>
      <th>brabancon_griffon</th>
      <th>briard</th>
      <th>brittany_spaniel</th>
      <th>bull_mastiff</th>
      <th>cairn</th>
      <th>cardigan</th>
      <th>chesapeake_bay_retriever</th>
      <th>chihuahua</th>
      <th>chow</th>
      <th>clumber</th>
      <th>cocker_spaniel</th>
      <th>collie</th>
      <th>curly-coated_retriever</th>
      <th>dandie_dinmont</th>
      <th>dhole</th>
      <th>dingo</th>
      <th>doberman</th>
      <th>...</th>
      <th>norwegian_elkhound</th>
      <th>norwich_terrier</th>
      <th>old_english_sheepdog</th>
      <th>otterhound</th>
      <th>papillon</th>
      <th>pekinese</th>
      <th>pembroke</th>
      <th>pomeranian</th>
      <th>pug</th>
      <th>redbone</th>
      <th>rhodesian_ridgeback</th>
      <th>rottweiler</th>
      <th>saint_bernard</th>
      <th>saluki</th>
      <th>samoyed</th>
      <th>schipperke</th>
      <th>scotch_terrier</th>
      <th>scottish_deerhound</th>
      <th>sealyham_terrier</th>
      <th>shetland_sheepdog</th>
      <th>shih-tzu</th>
      <th>siberian_husky</th>
      <th>silky_terrier</th>
      <th>soft-coated_wheaten_terrier</th>
      <th>staffordshire_bullterrier</th>
      <th>standard_poodle</th>
      <th>standard_schnauzer</th>
      <th>sussex_spaniel</th>
      <th>tibetan_mastiff</th>
      <th>tibetan_terrier</th>
      <th>toy_poodle</th>
      <th>toy_terrier</th>
      <th>vizsla</th>
      <th>walker_hound</th>
      <th>weimaraner</th>
      <th>welsh_springer_spaniel</th>
      <th>west_highland_white_terrier</th>
      <th>whippet</th>
      <th>wire-haired_fox_terrier</th>
      <th>yorkshire_terrier</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 121 columns</p>
</div>




```python
# append test image ID to predictions DataFrame
test_ids=[os.path.splitext(path)[0] for path in os.listdir(test_path)]
test_ids[:10]
```




    ['e334f758c7944df19c98d49498d28c64',
     'e19900119c7d3ce48d4035cd0211be72',
     'e6a5c18da7beedb1622bf7d18b452121',
     'dd3c80cee38d165aaf48083f4a4a0071',
     'df0d6ba158287cb2b3ed6459a22d42ba',
     'e37d651b9b5fdcf26ab37259fac877d1',
     'e0eea374c5170e90dc0b1ee795470ca6',
     'de2c6675c001726a96ad9d72ab229f3a',
     'e646ac89e0832502f9a726c72773cfcc',
     'e079440ee0061b92ec22faf17be13908']




```python
test_path
```




    '/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/'




```python
os.path.splitext(test_filenames[0])
```




    ('/content/drive/MyDrive/Colab Notebooks/Dog Vision/test/e334f758c7944df19c98d49498d28c64',
     '.jpg')




```python
preds_df['id']=test_ids
```


```python
preds_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>affenpinscher</th>
      <th>afghan_hound</th>
      <th>african_hunting_dog</th>
      <th>airedale</th>
      <th>american_staffordshire_terrier</th>
      <th>appenzeller</th>
      <th>australian_terrier</th>
      <th>basenji</th>
      <th>basset</th>
      <th>beagle</th>
      <th>bedlington_terrier</th>
      <th>bernese_mountain_dog</th>
      <th>black-and-tan_coonhound</th>
      <th>blenheim_spaniel</th>
      <th>bloodhound</th>
      <th>bluetick</th>
      <th>border_collie</th>
      <th>border_terrier</th>
      <th>borzoi</th>
      <th>boston_bull</th>
      <th>bouvier_des_flandres</th>
      <th>boxer</th>
      <th>brabancon_griffon</th>
      <th>briard</th>
      <th>brittany_spaniel</th>
      <th>bull_mastiff</th>
      <th>cairn</th>
      <th>cardigan</th>
      <th>chesapeake_bay_retriever</th>
      <th>chihuahua</th>
      <th>chow</th>
      <th>clumber</th>
      <th>cocker_spaniel</th>
      <th>collie</th>
      <th>curly-coated_retriever</th>
      <th>dandie_dinmont</th>
      <th>dhole</th>
      <th>dingo</th>
      <th>doberman</th>
      <th>...</th>
      <th>norwegian_elkhound</th>
      <th>norwich_terrier</th>
      <th>old_english_sheepdog</th>
      <th>otterhound</th>
      <th>papillon</th>
      <th>pekinese</th>
      <th>pembroke</th>
      <th>pomeranian</th>
      <th>pug</th>
      <th>redbone</th>
      <th>rhodesian_ridgeback</th>
      <th>rottweiler</th>
      <th>saint_bernard</th>
      <th>saluki</th>
      <th>samoyed</th>
      <th>schipperke</th>
      <th>scotch_terrier</th>
      <th>scottish_deerhound</th>
      <th>sealyham_terrier</th>
      <th>shetland_sheepdog</th>
      <th>shih-tzu</th>
      <th>siberian_husky</th>
      <th>silky_terrier</th>
      <th>soft-coated_wheaten_terrier</th>
      <th>staffordshire_bullterrier</th>
      <th>standard_poodle</th>
      <th>standard_schnauzer</th>
      <th>sussex_spaniel</th>
      <th>tibetan_mastiff</th>
      <th>tibetan_terrier</th>
      <th>toy_poodle</th>
      <th>toy_terrier</th>
      <th>vizsla</th>
      <th>walker_hound</th>
      <th>weimaraner</th>
      <th>welsh_springer_spaniel</th>
      <th>west_highland_white_terrier</th>
      <th>whippet</th>
      <th>wire-haired_fox_terrier</th>
      <th>yorkshire_terrier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e334f758c7944df19c98d49498d28c64</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e19900119c7d3ce48d4035cd0211be72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e6a5c18da7beedb1622bf7d18b452121</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dd3c80cee38d165aaf48083f4a4a0071</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>df0d6ba158287cb2b3ed6459a22d42ba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 121 columns</p>
</div>




```python
# add the prediction probabilities to each dog breed column
preds_df[list(unique_breeds)]=test_predictions
preds_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>affenpinscher</th>
      <th>afghan_hound</th>
      <th>african_hunting_dog</th>
      <th>airedale</th>
      <th>american_staffordshire_terrier</th>
      <th>appenzeller</th>
      <th>australian_terrier</th>
      <th>basenji</th>
      <th>basset</th>
      <th>beagle</th>
      <th>bedlington_terrier</th>
      <th>bernese_mountain_dog</th>
      <th>black-and-tan_coonhound</th>
      <th>blenheim_spaniel</th>
      <th>bloodhound</th>
      <th>bluetick</th>
      <th>border_collie</th>
      <th>border_terrier</th>
      <th>borzoi</th>
      <th>boston_bull</th>
      <th>bouvier_des_flandres</th>
      <th>boxer</th>
      <th>brabancon_griffon</th>
      <th>briard</th>
      <th>brittany_spaniel</th>
      <th>bull_mastiff</th>
      <th>cairn</th>
      <th>cardigan</th>
      <th>chesapeake_bay_retriever</th>
      <th>chihuahua</th>
      <th>chow</th>
      <th>clumber</th>
      <th>cocker_spaniel</th>
      <th>collie</th>
      <th>curly-coated_retriever</th>
      <th>dandie_dinmont</th>
      <th>dhole</th>
      <th>dingo</th>
      <th>doberman</th>
      <th>...</th>
      <th>norwegian_elkhound</th>
      <th>norwich_terrier</th>
      <th>old_english_sheepdog</th>
      <th>otterhound</th>
      <th>papillon</th>
      <th>pekinese</th>
      <th>pembroke</th>
      <th>pomeranian</th>
      <th>pug</th>
      <th>redbone</th>
      <th>rhodesian_ridgeback</th>
      <th>rottweiler</th>
      <th>saint_bernard</th>
      <th>saluki</th>
      <th>samoyed</th>
      <th>schipperke</th>
      <th>scotch_terrier</th>
      <th>scottish_deerhound</th>
      <th>sealyham_terrier</th>
      <th>shetland_sheepdog</th>
      <th>shih-tzu</th>
      <th>siberian_husky</th>
      <th>silky_terrier</th>
      <th>soft-coated_wheaten_terrier</th>
      <th>staffordshire_bullterrier</th>
      <th>standard_poodle</th>
      <th>standard_schnauzer</th>
      <th>sussex_spaniel</th>
      <th>tibetan_mastiff</th>
      <th>tibetan_terrier</th>
      <th>toy_poodle</th>
      <th>toy_terrier</th>
      <th>vizsla</th>
      <th>walker_hound</th>
      <th>weimaraner</th>
      <th>welsh_springer_spaniel</th>
      <th>west_highland_white_terrier</th>
      <th>whippet</th>
      <th>wire-haired_fox_terrier</th>
      <th>yorkshire_terrier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e334f758c7944df19c98d49498d28c64</td>
      <td>2.8746e-06</td>
      <td>2.95269e-09</td>
      <td>9.77277e-09</td>
      <td>9.74725e-09</td>
      <td>4.40952e-08</td>
      <td>8.7188e-09</td>
      <td>1.52999e-05</td>
      <td>5.89294e-08</td>
      <td>1.18975e-06</td>
      <td>1.21127e-07</td>
      <td>2.95217e-06</td>
      <td>2.80656e-07</td>
      <td>1.79893e-07</td>
      <td>2.26194e-09</td>
      <td>7.06435e-10</td>
      <td>9.47965e-08</td>
      <td>6.85876e-10</td>
      <td>1.69806e-08</td>
      <td>4.00444e-09</td>
      <td>7.54609e-10</td>
      <td>5.36746e-09</td>
      <td>4.16807e-08</td>
      <td>3.23415e-08</td>
      <td>1.09541e-08</td>
      <td>1.13463e-08</td>
      <td>5.54781e-09</td>
      <td>5.67186e-06</td>
      <td>1.13663e-08</td>
      <td>2.98207e-09</td>
      <td>1.64306e-07</td>
      <td>8.74198e-09</td>
      <td>4.14793e-07</td>
      <td>1.24214e-07</td>
      <td>5.75313e-10</td>
      <td>2.48675e-09</td>
      <td>0.975118</td>
      <td>1.48664e-09</td>
      <td>1.95483e-10</td>
      <td>2.59715e-10</td>
      <td>...</td>
      <td>1.2749e-10</td>
      <td>2.51515e-08</td>
      <td>8.91077e-08</td>
      <td>1.6216e-06</td>
      <td>7.40966e-09</td>
      <td>3.49403e-07</td>
      <td>3.27788e-08</td>
      <td>7.4159e-08</td>
      <td>3.72e-09</td>
      <td>2.31623e-07</td>
      <td>1.28485e-08</td>
      <td>1.28152e-09</td>
      <td>2.13609e-11</td>
      <td>7.09216e-09</td>
      <td>5.99683e-08</td>
      <td>4.13197e-08</td>
      <td>1.33571e-09</td>
      <td>5.29654e-06</td>
      <td>0.000224767</td>
      <td>1.32365e-10</td>
      <td>0.000102829</td>
      <td>5.08966e-10</td>
      <td>4.66317e-05</td>
      <td>1.23992e-07</td>
      <td>9.3648e-08</td>
      <td>4.30951e-09</td>
      <td>4.17241e-08</td>
      <td>1.74475e-08</td>
      <td>2.69169e-08</td>
      <td>2.61425e-06</td>
      <td>0.000264636</td>
      <td>3.25459e-06</td>
      <td>6.99907e-07</td>
      <td>2.51465e-08</td>
      <td>5.61051e-08</td>
      <td>2.85433e-09</td>
      <td>6.20245e-05</td>
      <td>1.5116e-09</td>
      <td>0.000136109</td>
      <td>1.30031e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e19900119c7d3ce48d4035cd0211be72</td>
      <td>1.72946e-13</td>
      <td>2.61498e-12</td>
      <td>2.5295e-13</td>
      <td>2.98609e-11</td>
      <td>2.5125e-09</td>
      <td>3.29329e-10</td>
      <td>2.11534e-11</td>
      <td>4.16218e-11</td>
      <td>4.94587e-08</td>
      <td>1.48477e-05</td>
      <td>4.51817e-13</td>
      <td>8.46452e-13</td>
      <td>1.7572e-12</td>
      <td>7.65779e-10</td>
      <td>3.14653e-10</td>
      <td>4.31209e-10</td>
      <td>3.99916e-12</td>
      <td>3.96274e-08</td>
      <td>2.03283e-12</td>
      <td>1.72363e-10</td>
      <td>5.16225e-12</td>
      <td>4.80491e-06</td>
      <td>1.0913e-10</td>
      <td>8.42611e-14</td>
      <td>3.7288e-07</td>
      <td>1.34519e-07</td>
      <td>1.52174e-13</td>
      <td>2.9558e-09</td>
      <td>3.51218e-10</td>
      <td>5.1261e-13</td>
      <td>1.80659e-11</td>
      <td>1.22327e-09</td>
      <td>4.10841e-11</td>
      <td>5.25073e-11</td>
      <td>1.98251e-13</td>
      <td>5.25202e-10</td>
      <td>1.2682e-11</td>
      <td>6.64333e-10</td>
      <td>5.48046e-12</td>
      <td>...</td>
      <td>5.78646e-11</td>
      <td>1.74336e-15</td>
      <td>9.68966e-12</td>
      <td>1.25215e-12</td>
      <td>4.6985e-09</td>
      <td>1.63514e-12</td>
      <td>5.26095e-10</td>
      <td>2.56796e-15</td>
      <td>1.73357e-12</td>
      <td>4.13643e-12</td>
      <td>2.59773e-11</td>
      <td>1.05882e-10</td>
      <td>0.999978</td>
      <td>5.11385e-11</td>
      <td>1.30721e-14</td>
      <td>4.52451e-12</td>
      <td>2.33866e-14</td>
      <td>1.47584e-14</td>
      <td>4.58389e-10</td>
      <td>2.82511e-10</td>
      <td>7.23282e-11</td>
      <td>5.78309e-11</td>
      <td>3.32814e-10</td>
      <td>2.63619e-08</td>
      <td>1.0164e-14</td>
      <td>1.55452e-12</td>
      <td>3.15379e-11</td>
      <td>2.58314e-13</td>
      <td>1.22441e-11</td>
      <td>5.69084e-10</td>
      <td>9.97393e-12</td>
      <td>3.10031e-11</td>
      <td>6.10504e-11</td>
      <td>3.89703e-11</td>
      <td>1.66105e-14</td>
      <td>1.55501e-06</td>
      <td>1.63989e-13</td>
      <td>4.52893e-10</td>
      <td>2.12586e-11</td>
      <td>3.48569e-13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e6a5c18da7beedb1622bf7d18b452121</td>
      <td>1.97527e-10</td>
      <td>1.88702e-10</td>
      <td>1.68102e-11</td>
      <td>9.74476e-11</td>
      <td>6.15706e-10</td>
      <td>1.90476e-12</td>
      <td>1.04248e-10</td>
      <td>2.33638e-12</td>
      <td>7.56912e-11</td>
      <td>3.87382e-12</td>
      <td>4.1726e-10</td>
      <td>1.32167e-10</td>
      <td>2.90295e-10</td>
      <td>1.12979e-06</td>
      <td>1.61364e-09</td>
      <td>1.42919e-10</td>
      <td>8.46685e-12</td>
      <td>1.88877e-10</td>
      <td>4.61848e-09</td>
      <td>3.16627e-09</td>
      <td>5.1182e-10</td>
      <td>3.23332e-11</td>
      <td>3.11174e-10</td>
      <td>4.50236e-11</td>
      <td>3.52546e-09</td>
      <td>7.08177e-12</td>
      <td>1.48956e-10</td>
      <td>3.1909e-10</td>
      <td>7.0167e-09</td>
      <td>2.70773e-10</td>
      <td>5.95204e-11</td>
      <td>2.78229e-08</td>
      <td>1.88792e-06</td>
      <td>5.0243e-09</td>
      <td>0.000126616</td>
      <td>9.13435e-11</td>
      <td>2.69665e-10</td>
      <td>6.75046e-11</td>
      <td>9.76503e-11</td>
      <td>...</td>
      <td>5.68158e-10</td>
      <td>1.88542e-09</td>
      <td>6.85099e-11</td>
      <td>1.39066e-10</td>
      <td>2.03311e-11</td>
      <td>2.27895e-10</td>
      <td>4.53951e-10</td>
      <td>2.13164e-09</td>
      <td>1.00501e-09</td>
      <td>1.30191e-09</td>
      <td>5.31929e-12</td>
      <td>4.00198e-11</td>
      <td>1.83575e-10</td>
      <td>1.74617e-10</td>
      <td>1.63068e-10</td>
      <td>1.11255e-09</td>
      <td>1.11556e-07</td>
      <td>1.21721e-10</td>
      <td>2.51856e-11</td>
      <td>1.51682e-10</td>
      <td>1.76108e-11</td>
      <td>2.72146e-10</td>
      <td>9.12017e-11</td>
      <td>2.59142e-12</td>
      <td>1.95614e-09</td>
      <td>8.02078e-07</td>
      <td>9.11691e-11</td>
      <td>0.000108095</td>
      <td>1.0591e-10</td>
      <td>1.55785e-10</td>
      <td>5.09907e-09</td>
      <td>1.3655e-11</td>
      <td>3.20019e-10</td>
      <td>9.36478e-08</td>
      <td>7.5274e-11</td>
      <td>4.01126e-09</td>
      <td>2.72734e-11</td>
      <td>2.61903e-10</td>
      <td>3.44375e-12</td>
      <td>2.825e-09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dd3c80cee38d165aaf48083f4a4a0071</td>
      <td>3.88313e-06</td>
      <td>8.87407e-10</td>
      <td>5.08484e-12</td>
      <td>8.91422e-11</td>
      <td>9.86823e-12</td>
      <td>3.89629e-13</td>
      <td>5.44845e-15</td>
      <td>2.43478e-11</td>
      <td>6.09393e-12</td>
      <td>5.76128e-15</td>
      <td>2.29998e-11</td>
      <td>4.52755e-11</td>
      <td>9.70081e-14</td>
      <td>1.36788e-09</td>
      <td>2.85764e-12</td>
      <td>2.96295e-14</td>
      <td>1.17312e-11</td>
      <td>4.37679e-13</td>
      <td>9.40563e-14</td>
      <td>2.51177e-13</td>
      <td>0.999828</td>
      <td>1.48199e-13</td>
      <td>4.26564e-11</td>
      <td>2.44939e-07</td>
      <td>4.8841e-12</td>
      <td>3.41381e-08</td>
      <td>1.10208e-09</td>
      <td>1.01481e-13</td>
      <td>9.2117e-13</td>
      <td>1.11092e-14</td>
      <td>4.03475e-10</td>
      <td>6.12491e-14</td>
      <td>3.17955e-09</td>
      <td>5.86763e-14</td>
      <td>6.40514e-11</td>
      <td>2.55675e-09</td>
      <td>5.72465e-13</td>
      <td>1.82571e-12</td>
      <td>1.43845e-13</td>
      <td>...</td>
      <td>1.4143e-12</td>
      <td>4.2412e-12</td>
      <td>1.32263e-08</td>
      <td>2.17925e-08</td>
      <td>2.49932e-14</td>
      <td>2.59952e-10</td>
      <td>6.53012e-16</td>
      <td>6.41203e-13</td>
      <td>6.14635e-10</td>
      <td>3.05245e-12</td>
      <td>9.13857e-14</td>
      <td>1.42095e-11</td>
      <td>3.6216e-11</td>
      <td>1.53656e-12</td>
      <td>4.56721e-11</td>
      <td>3.40063e-08</td>
      <td>3.06028e-11</td>
      <td>9.17577e-10</td>
      <td>9.2776e-12</td>
      <td>1.39253e-13</td>
      <td>2.78088e-11</td>
      <td>1.08164e-10</td>
      <td>7.63946e-13</td>
      <td>1.71215e-06</td>
      <td>2.20781e-11</td>
      <td>1.7203e-09</td>
      <td>2.43708e-08</td>
      <td>6.09385e-09</td>
      <td>2.26379e-08</td>
      <td>0.000147634</td>
      <td>4.96061e-07</td>
      <td>1.77229e-13</td>
      <td>9.30376e-12</td>
      <td>2.48676e-12</td>
      <td>1.00661e-11</td>
      <td>5.07824e-16</td>
      <td>5.30792e-16</td>
      <td>4.24784e-11</td>
      <td>2.64864e-12</td>
      <td>3.78027e-12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>df0d6ba158287cb2b3ed6459a22d42ba</td>
      <td>7.08184e-05</td>
      <td>2.91845e-08</td>
      <td>2.23287e-06</td>
      <td>1.2348e-05</td>
      <td>0.00102457</td>
      <td>3.66416e-07</td>
      <td>7.1959e-07</td>
      <td>3.22634e-07</td>
      <td>4.40434e-09</td>
      <td>2.23796e-08</td>
      <td>1.24937e-09</td>
      <td>2.8615e-10</td>
      <td>1.21093e-06</td>
      <td>1.25435e-10</td>
      <td>8.34051e-09</td>
      <td>1.57847e-08</td>
      <td>3.50274e-07</td>
      <td>6.38442e-07</td>
      <td>4.66723e-10</td>
      <td>0.0042907</td>
      <td>5.44724e-05</td>
      <td>3.96228e-07</td>
      <td>2.57397e-05</td>
      <td>2.86402e-09</td>
      <td>2.24241e-09</td>
      <td>3.54757e-05</td>
      <td>3.94109e-07</td>
      <td>1.63653e-05</td>
      <td>5.82148e-08</td>
      <td>1.11559e-05</td>
      <td>1.2159e-09</td>
      <td>1.85084e-10</td>
      <td>2.41501e-08</td>
      <td>3.46738e-06</td>
      <td>6.81973e-06</td>
      <td>4.48235e-11</td>
      <td>6.93228e-08</td>
      <td>0.00121963</td>
      <td>0.151396</td>
      <td>...</td>
      <td>0.00041117</td>
      <td>8.10868e-07</td>
      <td>3.57073e-09</td>
      <td>6.32433e-10</td>
      <td>1.17676e-06</td>
      <td>1.12751e-07</td>
      <td>6.53507e-10</td>
      <td>1.96087e-08</td>
      <td>4.15432e-06</td>
      <td>0.000114165</td>
      <td>3.16978e-07</td>
      <td>1.10448e-08</td>
      <td>1.20362e-10</td>
      <td>8.4058e-09</td>
      <td>5.44487e-07</td>
      <td>0.514346</td>
      <td>0.00113495</td>
      <td>1.75748e-07</td>
      <td>1.93334e-05</td>
      <td>4.79962e-08</td>
      <td>6.93062e-09</td>
      <td>1.16143e-06</td>
      <td>4.73505e-06</td>
      <td>3.00507e-06</td>
      <td>0.0112202</td>
      <td>6.11402e-08</td>
      <td>3.46682e-05</td>
      <td>5.57502e-08</td>
      <td>1.07794e-07</td>
      <td>8.44742e-08</td>
      <td>2.36538e-07</td>
      <td>0.00317715</td>
      <td>1.20292e-06</td>
      <td>8.84313e-07</td>
      <td>2.22874e-06</td>
      <td>1.08701e-09</td>
      <td>1.65078e-06</td>
      <td>1.40521e-07</td>
      <td>1.18312e-05</td>
      <td>3.28496e-08</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 121 columns</p>
</div>




```python
# export for submission
preds_df.to_csv('/content/drive/MyDrive/Colab Notebooks/Dog Vision/full_model_predictions_submission_1_mobilenetV2.csv',
                index=False)
```

## Making prediction on custom images

* Get the filepaths of images
* Turn the filepaths into data batches using `create_data_batches()` and set the `test_data` parameter to `True`
* Pass the custom image data batch to our model's `predict()` method
* Convert the prediction output probabilities to prediction labels
* Compare the predicted labels to the custom images


```python
custom_path='/content/drive/MyDrive/Colab Notebooks/Dog Vision/dog photos/'
custom_image_paths=[custom_path+fname for fname in os.listdir(custom_path)]
```


```python
custom_image_paths
```




    ['/content/drive/MyDrive/Colab Notebooks/Dog Vision/dog photos/dog-photo-2.jpeg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/dog photos/dog-photo-1.jpeg',
     '/content/drive/MyDrive/Colab Notebooks/Dog Vision/dog photos/dog-photo-3.jpeg']




```python
# turn custom images into batch dataset
custom_data=create_data_batches(custom_image_paths,test_data=True)
custom_data
```

    Creating test data batches...





    <BatchDataset shapes: (None, 224, 224, 3), types: tf.float32>




```python
# Make predictions
custom_preds=loaded_full_model.predict(custom_data)
```


```python
custom_preds.shape
```




    (3, 120)




```python
custom_preds
```




    array([[1.09000664e-09, 8.09645684e-10, 1.97352291e-11, 1.61102590e-10,
            1.08583856e-06, 2.25018823e-10, 3.18582039e-07, 4.55783356e-06,
            2.36346253e-07, 1.60012496e-05, 6.10213320e-08, 5.85708782e-13,
            5.63054256e-11, 3.42836037e-10, 1.48426702e-08, 4.02748584e-10,
            1.11434528e-09, 5.68220969e-07, 8.46975046e-10, 1.01812491e-11,
            2.04216661e-11, 1.34442516e-07, 1.52628559e-08, 7.65897623e-11,
            1.27564059e-09, 5.93115423e-08, 1.63675349e-08, 4.27420787e-07,
            1.51903592e-07, 7.80349001e-06, 3.14522586e-09, 6.93694288e-13,
            5.64916991e-04, 1.38904621e-09, 8.77800872e-08, 6.48188059e-10,
            3.25500564e-11, 2.05587469e-07, 4.04990069e-11, 1.00608399e-09,
            2.97737444e-07, 1.69453187e-06, 9.28861571e-11, 2.53274202e-09,
            3.96549012e-07, 8.35621563e-08, 3.49967277e-11, 5.46409495e-09,
            3.29206384e-10, 1.33081034e-01, 8.64335288e-06, 1.39955284e-06,
            9.27463972e-09, 2.53774317e-08, 5.12237352e-16, 5.94383209e-10,
            1.25069310e-09, 1.83502721e-06, 5.12597506e-14, 2.34337605e-10,
            1.91281324e-09, 3.43752370e-12, 4.79931612e-14, 3.93123395e-10,
            1.82733516e-11, 1.04055387e-09, 5.26966137e-10, 1.46404237e-01,
            7.10929692e-01, 9.30081220e-11, 6.27786036e-08, 6.67567002e-10,
            2.95988944e-09, 8.22032309e-09, 6.72520303e-11, 2.11191627e-06,
            2.22643894e-06, 1.23076425e-05, 6.06370856e-11, 1.01241785e-04,
            7.86961785e-09, 2.14067555e-10, 2.75716200e-10, 4.32338894e-08,
            5.30912536e-11, 1.30875407e-07, 5.21793737e-08, 2.97797076e-09,
            4.81715006e-06, 4.25830962e-08, 1.90809701e-10, 3.38638367e-10,
            7.27008564e-10, 2.93228413e-06, 7.00649131e-12, 7.64437125e-10,
            1.40003675e-09, 3.36909967e-09, 1.68622383e-09, 2.83213936e-14,
            6.91977107e-07, 8.53082511e-08, 2.37660776e-11, 2.94564506e-06,
            2.78359007e-08, 7.02022662e-09, 1.58512689e-10, 1.42964859e-11,
            6.99338070e-12, 6.44737666e-11, 2.06892801e-04, 8.62641074e-03,
            9.95142727e-06, 1.46930816e-07, 2.60367172e-09, 2.03891778e-10,
            1.09298473e-11, 2.26484886e-07, 1.24651227e-11, 5.62885191e-07],
           [2.35636094e-10, 1.98572137e-09, 1.80582944e-11, 1.23305410e-09,
            3.23054974e-06, 5.22342889e-05, 2.53988883e-08, 3.68914996e-08,
            2.42141649e-08, 1.01593002e-06, 8.51060472e-11, 3.62050251e-11,
            1.70314743e-05, 1.81216038e-08, 8.50403012e-06, 1.87224498e-06,
            3.68404881e-06, 4.16453139e-09, 2.01904641e-06, 9.18584135e-07,
            7.13600923e-12, 2.40693354e-09, 2.43600375e-08, 2.06686046e-09,
            1.60070044e-07, 7.93136191e-04, 4.22148075e-13, 1.65516951e-08,
            1.17710715e-05, 2.61140283e-08, 9.48217746e-07, 9.92339544e-10,
            2.49536856e-06, 1.40253163e-04, 3.93686306e-10, 1.36909819e-07,
            3.16335402e-09, 2.77518720e-05, 1.01883973e-06, 2.93053034e-07,
            1.97700774e-06, 2.81089929e-10, 4.51896369e-08, 2.19947906e-04,
            1.84775854e-05, 1.69084036e-08, 9.27412032e-07, 9.61093316e-10,
            1.70934928e-08, 9.79479909e-01, 6.45833950e-07, 6.01253065e-04,
            2.30255377e-04, 6.59908494e-07, 1.13143912e-08, 2.07188312e-11,
            4.19668211e-09, 2.31340147e-09, 1.85500335e-12, 3.96129417e-06,
            3.37721384e-09, 1.24411897e-10, 6.00155039e-12, 1.01136146e-04,
            6.97189227e-14, 9.23314580e-09, 8.27349254e-07, 9.90883913e-03,
            1.49637617e-05, 2.53755170e-05, 7.20035753e-10, 1.90278240e-06,
            7.40328005e-07, 5.70625734e-11, 2.17791910e-10, 1.48432544e-10,
            1.17324822e-11, 4.25905179e-11, 3.46344882e-06, 1.26528887e-09,
            3.73320486e-09, 4.27225061e-10, 6.95085794e-11, 3.13169202e-09,
            1.97703383e-08, 2.19961183e-09, 1.49307889e-05, 6.63510535e-09,
            1.38689054e-03, 3.86639886e-06, 2.86284012e-06, 5.30039069e-05,
            1.45674939e-03, 3.12142656e-03, 2.28608087e-11, 2.35479747e-10,
            5.90110807e-11, 1.46657442e-09, 2.71070143e-14, 1.19878880e-08,
            1.09425329e-07, 2.07783330e-07, 2.75922389e-07, 1.28781541e-09,
            2.16408669e-08, 6.32185856e-06, 4.53594708e-13, 2.29511485e-08,
            5.87932254e-06, 1.71369706e-07, 1.94966265e-09, 3.30693095e-09,
            2.44667202e-08, 1.99147942e-03, 2.71407538e-04, 6.18353937e-08,
            7.19168952e-12, 6.15587368e-08, 3.01359759e-09, 3.11032415e-11],
           [3.96088135e-10, 2.02269046e-08, 1.05097187e-09, 1.36528172e-07,
            6.56929251e-06, 1.86596383e-09, 5.72626557e-10, 5.79060361e-05,
            1.39340159e-06, 7.62142532e-04, 6.52505463e-08, 2.13585913e-10,
            3.05077151e-06, 1.03677415e-07, 7.78063259e-06, 8.28121131e-07,
            6.88821400e-09, 4.57799274e-07, 7.16109025e-06, 4.26401385e-07,
            1.04018749e-08, 2.38567463e-06, 1.79512199e-07, 2.71778745e-11,
            9.80738122e-08, 1.94902539e-01, 2.21990922e-06, 3.05851898e-07,
            6.08236296e-03, 5.58463871e-05, 3.39766091e-04, 1.71625878e-08,
            6.84116048e-06, 3.62583160e-05, 1.25048231e-07, 9.67490905e-08,
            1.96413355e-04, 4.88506518e-02, 3.25369527e-08, 9.03144337e-06,
            3.68156838e-10, 9.10933196e-09, 2.20073653e-05, 1.53498736e-03,
            1.37684683e-05, 1.73435465e-03, 2.94483965e-04, 1.34837990e-07,
            3.42419497e-08, 3.71050253e-03, 1.66737959e-06, 3.70014488e-04,
            1.86668581e-06, 5.28105484e-05, 1.99329335e-07, 2.31039916e-08,
            9.54105017e-10, 7.55186278e-08, 3.33165495e-08, 7.18793808e-06,
            2.34384501e-07, 3.93146522e-11, 2.00848609e-11, 2.01374743e-04,
            7.60819574e-10, 4.59558441e-06, 2.37276345e-01, 2.79496402e-01,
            9.17872530e-04, 1.00350985e-06, 2.26924115e-08, 1.33479713e-04,
            1.91552273e-04, 1.88157365e-10, 1.87548616e-10, 6.30937211e-06,
            7.82258326e-07, 1.92142784e-06, 3.52074693e-07, 1.70225647e-08,
            7.09665692e-05, 1.77326783e-05, 6.46137518e-11, 2.32704900e-09,
            6.72696785e-07, 3.60205310e-09, 1.02428712e-05, 1.53889845e-09,
            1.34143233e-01, 1.35327014e-07, 2.34370759e-06, 9.45430729e-05,
            6.86950589e-05, 5.40122874e-02, 1.90208449e-09, 1.63602394e-07,
            1.02289789e-07, 4.04588230e-09, 3.08199439e-08, 1.61988686e-11,
            4.92808795e-07, 3.37395296e-02, 1.72839023e-08, 5.35881991e-05,
            1.45881506e-06, 2.75164239e-05, 3.09023196e-10, 5.87447602e-09,
            3.56087776e-06, 2.17101537e-08, 4.70910976e-07, 1.27347857e-05,
            4.61918108e-11, 1.52215689e-05, 3.58982265e-06, 2.27806829e-10,
            8.01025524e-10, 4.07222105e-04, 3.76004277e-06, 9.57045465e-09]],
          dtype=float32)




```python
custom_preds_labels=[get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_preds_labels
```




    ['lakeland_terrier', 'golden_retriever', 'labrador_retriever']




```python
# get custom images 
custom_images=[]

for image in custom_data.unbatch().as_numpy_iterator():
  custom_images.append(image)
```


```python
#custom_images
plt.figure(figsize=(10,10))
for i,image in enumerate(custom_images):
  plt.subplot(1,3,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.title(custom_preds_labels[i])
  plt.imshow(image)
```


    
![png](output_168_0.png)
    



```python

```
