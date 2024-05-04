import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, Activation, Concatenate, Layer, UpSampling2D, Conv2DTranspose, \
    ZeroPadding2D, ReLU, LeakyReLU, ELU
from keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.applications import VGG16


class ReconstructionLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.losses.MeanAbsoluteError()

    def call(self, results, targets):
        loss = 0.
        for res, target in zip(results, targets):
            loss += self.l1(res, target)
        return loss / tf.cast(tf.size(results), tf.float32)


class VGGFeature(tf.keras.Model):
    def __init__(self):
        super().__init__()

        vgg16 = VGG16(include_top=False, weights='imagenet')
        vgg16.trainable = False

        self.vgg16_layers = vgg16.layers
        # Define the layers for feature extraction
        self.vgg16_pool_1 = tf.keras.Sequential(self.vgg16_layers[0:5])
        self.vgg16_pool_2 = tf.keras.Sequential(self.vgg16_layers[5:10])
        self.vgg16_pool_3 = tf.keras.Sequential(self.vgg16_layers[10:17])

    def call(self, x):
        # Extract features
        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)

        return [pool_1, pool_2, pool_3]


class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.l1loss = tf.keras.losses.MeanAbsoluteError()

    def call(self, vgg_results, vgg_targets):
        loss = 0.
        for vgg_res, vgg_target in zip(vgg_results, vgg_targets):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(feat_res, feat_target)
        return loss / tf.cast(tf.size(vgg_results), dtype=tf.float32)

class StyleLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.l1loss = tf.keras.losses.MeanAbsoluteError()

    def gram(self, feature):
        shape = tf.shape(feature)
        n, c, h, w = shape[0], shape[1], shape[2], shape[3]
        feature = tf.reshape(feature, (n, c, h * w))
        gram_mat = tf.linalg.matmul(feature, feature, transpose_b=True)
        return gram_mat / tf.cast(c * h * w, tf.float32)

    def call(self, vgg_results, vgg_targets):
        loss = 0.
        for vgg_res, vgg_target in zip(vgg_results, vgg_targets):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(self.gram(feat_res), self.gram(feat_target))
        return loss / tf.cast(tf.size(vgg_results), tf.float32)
    

class TotalVariationLoss(tf.keras.losses.Loss):
    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img
        # Define the kernel for convolution
        kernel = tf.constant([
            [0, 1, 0],
            [1, -2, 0],
            [0, 0, 0]], dtype=tf.float32)
        kernel = tf.reshape(kernel, (1, 1, 3, 3))
        kernel = tf.tile(kernel, [c_img, 1, 1, 1])
        self.kernel = tf.Variable(kernel, trainable=False)

    def gradient(self, x):
        return tf.nn.depthwise_conv2d(
            x, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

    def call(self, results, mask):
        loss = 0.
        for res in results:
            # Resize mask to match the dimensions of the result
            resized_mask = tf.image.resize(mask, tf.shape(res)[1:3])
            # Calculate the gradient
            grad = self.gradient(res) * resized_mask
            # Compute the mean of the absolute values of the gradient
            loss += tf.reduce_mean(tf.abs(grad))
        return loss / tf.cast(tf.size(results), tf.float32)

class InpaintLoss(tf.keras.losses.Loss):
    def __init__(
            self, c_img=3, w_l1=6., w_percep=0.1, w_style=240., w_tv=0.1,
            structure_layers=[0, 1, 2, 3, 4, 5],
            texture_layers=[0, 1, 2]):
        super().__init__()

        self.l_struct = structure_layers
        self.l_text = texture_layers

        self.w_l1 = w_l1
        self.w_percep = w_percep
        self.w_style = w_style
        self.w_tv = w_tv

        # Initialize the loss components with their respective TensorFlow versions
        self.reconstruction_loss = ReconstructionLoss()
        self.vgg_feature = VGGFeature()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss(c_img)

    # todo add mask
    def call(self, results, target):
        print(results.shape, target.shape)
        # Resize target to match the dimensions of the results
        targets = [tf.image.resize(target, tf.shape(res)[1:3]) for res in results]

        loss_struct = 0.
        loss_text = 0.
        loss_list = {}

        if len(self.l_struct) > 0:
            struct_r = [results[i] for i in self.l_struct]
            struct_t = [targets[i] for i in self.l_struct]

            # Calculate structural loss
            loss_struct = self.reconstruction_loss(struct_r, struct_t) * self.w_l1
            loss_list['reconstruction_loss'] = loss_struct.numpy()

        if len(self.l_text) > 0:
            text_r = [targets[i] for i in self.l_text]
            text_t = [results[i] for i in self.l_text]

            # Extract VGG features
            vgg_r = [self.vgg_feature(f) for f in text_r]
            vgg_t = [self.vgg_feature(t) for t in text_t]

            # Calculate style, perceptual, and total variation losses
            loss_style = self.style_loss(vgg_r, vgg_t) * self.w_style
            loss_percep = self.perceptual_loss(vgg_r, vgg_t) * self.w_percep
            loss_tv = self.tv_loss(text_r, mask) * self.w_tv

            loss_text = loss_style + loss_percep + loss_tv
            loss_list.update({
                'perceptual_loss': loss_percep.numpy(),
                'style_loss': loss_style.numpy(),
                'total_variation_loss': loss_tv.numpy()
            })

        loss_total = loss_struct + loss_text

        return loss_total, loss_list


class BlendBlock(Layer):
    def __init__(self, c_in, c_out, ksize_mid=3, norm='batch', act='relu'):
        super(BlendBlock, self).__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = Sequential([
            Conv2D(c_mid, 1, 1, padding='same'),
            Activation('relu'),
            Conv2D(c_out, ksize_mid, 1, padding='same'),
            Activation('relu'),
            Conv2D(c_out, 1, 1, padding='same'),
            Activation('sigmoid')
        ])

    def call(self, x, **kwargs):
        return self.blend(x)


class FusionBlock(Layer):
    def __init__(self, c_feat, c_alpha=1):
        super(FusionBlock, self).__init__()
        c_img = 3
        self.map2img = Sequential([
            Conv2D(c_img, 1, 1, padding='same'),
            Activation('sigmoid')
        ])
        self.blend = BlendBlock(c_img * 2, c_alpha)

    def call(self, img_miss, feat_de, **kwargs):
        img_miss = tf.image.resize(img_miss, feat_de.shape[1:3])
        raw = self.map2img(feat_de)
        alpha = self.blend(tf.concat([img_miss, raw], axis=3))
        result = alpha * raw + (1 - alpha) * img_miss
        return result, alpha, raw


class DecodeBlock(Layer):
    def __init__(self, c_from_up, c_from_down, c_out, mode='nearest',
                 kernel_size=4, scale=2, normalization='batch', activation='relu'):
        super(DecodeBlock, self).__init__()

        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpSampling2D(size=(scale, scale), interpolation=mode)

        layers = [Conv2D(self.c_out, kernel_size, strides=1, padding='same')]
        if normalization:
            # Assuming get_norm is defined elsewhere
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(Activation(activation))
        self.decode = Sequential(layers)

    def call(self, x, concat=None, **kwargs):
        out = self.up(x)
        if self.c_from_down > 0:
            out = Concatenate(axis=-1)([out, concat])
        out = self.decode(out)
        return out


class EncodeBlock(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super(EncodeBlock, self).__init__()

        self.c_in = in_channels
        self.c_out = out_channels

        layers = [Conv2D(self.c_out, kernel_size, strides=stride, padding='same')]
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(Activation(activation))
        self.encode = Sequential(layers)

    def call(self, x, **kwargs):
        return self.encode(x)


class UpBlock(Layer):
    def __init__(self, mode='nearest', scale=2, channel=None, kernel_size=4):
        super(UpBlock, self).__init__()

        self.mode = mode
        if mode == 'deconv':
            self.up = Conv2DTranspose(channel, kernel_size, strides=scale, padding='same')
        else:
            def upsample(x):
                return tf.image.resize(x, size=(x.shape[1] * scale, x.shape[2] * scale), method=mode)

            self.up = upsample

    def call(self, x, **kwargs):
        return self.up(x)


class ConvTranspose2dSame(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTranspose2dSame, self).__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)
        self.trans_conv = Conv2DTranspose(out_channels, kernel_size, strides=stride, padding='same',
                                          output_padding=output_padding)

    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def call(self, x):
        return self.trans_conv(x)



class Conv2dSame(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dSame, self).__init__()

        padding = self.conv_same_pad(kernel_size, stride)
        if isinstance(padding, int):
            self.conv = Conv2D(out_channels, kernel_size, strides=stride, padding='same')
        else:
            self.conv = Sequential([
                ZeroPadding2D(padding),
                Conv2D(out_channels, kernel_size, strides=stride, padding='valid')
            ])

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return (left, right)

    def call(self, x):
        return self.conv(x)


def get_activation(name):
    if name == 'relu':
        activation = ReLU()
    elif name == 'elu':
        activation = ELU()
    elif name == 'leaky_relu':
        activation = LeakyReLU(alpha=0.2)
    elif name == 'tanh':
        activation = Activation('tanh')
    elif name == 'sigmoid':
        activation = Activation('sigmoid')
    else:
        activation = None
    return activation


def get_norm(name, out_channels):
    if name == 'batch':
        norm = BatchNormalization()
    elif name == 'instance':
        norm = LayerNormalization()
    else:
        norm = None
    return norm


class DFNet(Model):
    def __init__(
            self, c_img=3, c_mask=1, c_alpha=3,
            mode='nearest', norm='batch', act_en='relu', act_de='relu',
            en_ksize=[7, 5, 5, 3, 3, 3, 3, 3], de_ksize=[3] * 8,
            blend_layers=[0, 1, 2, 3, 4, 5], **kwargs):
        super(DFNet, self).__init__(**kwargs)

        c_init = c_img + c_mask

        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        assert self.n_en == self.n_de, (
            'The number of layers in Encoder and Decoder must be equal.')
        assert self.n_en >= 1, (
            'The number of layers in Encoder and Decoder must be greater than 1.')

        assert 0 in blend_layers, 'Layer 0 must be blended.'

        self.en = []
        c_in = c_init
        self.en.append(
            EncodeBlock(c_in, 64, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            c_in = self.en[-1].c_out
            c_out = min(c_in * 2, 512)
            self.en.append(EncodeBlock(
                c_in, c_out, k_en, stride=2,
                normalization=norm, activation=act_en))

        self.de = []
        self.fuse = []
        for i, k_de in enumerate(de_ksize):
            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i - 1].c_in
            layer_idx = self.n_de - i - 1

            self.de.append(DecodeBlock(
                c_from_up, c_from_down, c_out, mode, k_de, scale=2,
                normalization=norm, activation=act_de))
            if layer_idx in blend_layers:
                self.fuse.append(FusionBlock(c_out, c_alpha))
            else:
                self.fuse.append(None)

        # register parameters
        for i, de in enumerate(self.de[::-1]):
            self.__setattr__('de_{}'.format(i), de)
        for i, fuse in enumerate(self.fuse[::-1]):
            if fuse:
                self.__setattr__('fuse_{}'.format(i), fuse)

    # img_miss_with_mask should be an array of [img_miss, mask]
    def call(self, img_miss_with_mask):
        print(tf.shape(img_miss_with_mask))
        img_miss = img_miss_with_mask[0]
        mask = img_miss_with_mask[1]
        out = tf.concat([img_miss, mask], axis=3)

        out_en = [out]
        for encode in self.en:
            out = encode(out)
            out_en.append(out)

        results = []
        alphas = []
        raws = []
        for i, (decode, fuse) in enumerate(zip(self.de, self.fuse)):
            out = decode(out, out_en[-i - 2])
            if fuse:
                result, alpha, raw = fuse(img_miss, out)
                results.append(result)
                alphas.append(alpha)
                raws.append(raw)

        print('Model output', tf.shape(results), tf.shape(alphas), tf.shape(raws))
        return results[::-1], alphas[::-1], raws[::-1]


# TODO: change the [image, image] array to [image, mask] array
def generate_masks_outputs(e):
    image = tf.cast(e['image'], tf.float16) / 255.
    mask = tf.cast(e['image'], tf.float16) / 255.

    return (
        [image, mask],
        image,
    )


EPOCHS = 1
BATCH_SIZE = 42

tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    x = tfds.load('places365_small', split='test', download=True)
    x = x.map(generate_masks_outputs)
    x = x.shuffle(512)
    x = x.batch(BATCH_SIZE)

    model = DFNet()
    model.compile(
        optimizer='adam',
        loss=InpaintLoss(),
    )

    model.fit(x, epochs=EPOCHS)
    model.summary()
