import json
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.optimizers
from keras import objectives
import keras.callbacks
from keras.callbacks import Callback, CallbackList
from .util           import ensure_list, NpEncoder, curry
from .util.tuning    import InvalidHyperparameterError
from .util.layers    import LinearSchedule
import os.path
import random
import matplotlib.pyplot as plt
import wandb
# modified version
import progressbar
class DynamicMessage(progressbar.DynamicMessage):
    def __call__(self, progress, data):
        val = data['dynamic_messages'][self.name]
        if val:
            return '{}'.format(val)
        else:
            return 6 * '-'



class Network:
    """Base class for various neural networks including GANs, AEs and Classifiers.
Provides an interface for saving / loading the trained weights as well as hyperparameters.

Each instance corresponds to a directory (specified in the `path` variable in the initialization),
which contains the learned weights as well as several json files for the metadata.
If a network depends on another network (i.e. AAE depends on SAE), it is customary to
save the network inside the dependent network. (e.g. Saving an AAE to mnist_SAE/_AAE)

PARAMETERS dict in the initialization argument is stored in the instance as well as 
serialized into a JSON string and is subsequently reloaded along with the weights.
This dict can be used while building the network, making it easier to perform a hyperparameter tuning.
"""
    def __init__(self,path,parameters={}):
        import subprocess
        subprocess.call(["mkdir","-p",path])
        self.built = False
        self.built_aux = False
        self.compiled = False
        self.loaded = False
        self.parameters = parameters
        self.custom_log_functions = {}
        self.metrics = []
        self.nets    = [None]
        self.losses  = [None]
        if parameters:
            # handle the test-time where parameters is not given
            self.path = os.path.join(path,"logs",self.parameters["time_start"])
            self.file_writer = tf.summary.FileWriter(self.path)
            self.epoch = LinearSchedule(schedule={
                0:0,
                self.parameters["epoch"]:self.parameters["epoch"],
            }, name="epoch")
            self.callbacks = [
                # keras.callbacks.LambdaCallback(
                #     on_epoch_end = self.save_epoch(path=self.path,freq=1000)),
                keras.callbacks.LambdaCallback(
                    on_epoch_end=self.epoch.update),
                keras.callbacks.TerminateOnNaN(),
                keras.callbacks.LambdaCallback(
                    on_epoch_end = self.bar_update,),
                # keras.callbacks.LambdaCallback(
                #     on_batch_end = self.bar_update,),
                keras.callbacks.TensorBoard(
                    histogram_freq = 0,
                    log_dir     = self.path,
                    write_graph = False)
            ]
        else:
            self.path = path
            self.callbacks = []

    def build(self,*args,**kwargs):
        """An interface for building a network. Input-shape: list of dimensions.
Users should not overload this method; Define _build_around() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built:
            # print("Avoided building {} twice.".format(self))
            return
        print("Building networks")
        self._build_around(*args,**kwargs)
        self.built = True
        print("Network built")
        return self

    def _build_around(self,*args,**kwargs):
        """An interface for building a network.
This function is called by build() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build_around() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        return self._build_primary(*args,**kwargs)

    def _build_primary(self,*args,**kwargs):
        pass

    def build_aux(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
Users should not overload this method; Define _build_around() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built_aux:
            # print("Avoided building {} twice.".format(self))
            return
        print("Building auxiliary networks")
        self._build_aux_around(*args,**kwargs)
        self.built_aux = True
        print("Auxiliary network built")
        return self

    def _build_aux_around(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
This function is called by build_aux() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build_aux_around() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        return self._build_aux_primary(*args,**kwargs)

    def _build_aux_primary(self,*args,**kwargs):
        pass

    def compile(self,*args,**kwargs):
        """An interface for compiling a network."""
        if self.compiled:
            # print("Avoided compiling {} twice.".format(self))
            return
        print("Compiling networks")
        self._compile(*args,**kwargs)
        self.compiled = True
        self.loaded = True
        print("Network compiled")
        return self

    def _compile(self,optimizers):
        """An interface for compileing a network."""
        # default method.
        print(f"there are {len(self.nets)} networks.")
        print(f"there are {len(optimizers)} optimizers.")
        print(f"there are {len(self.losses)} losses.")
        assert len(self.nets) == len(optimizers)
        assert len(self.nets) == len(self.losses)
        for net, o, loss in zip(self.nets, optimizers, self.losses):
            print(f"compiling {net} with {o}, {loss}.")
            net.compile(optimizer=o, loss=loss, metrics=self.metrics)
        return

    def local(self,path=""):
        """A convenient method for converting a relative path to the learned result directory
into a full path."""
        return os.path.join(self.path,path)



    def _save(self, path="", epoch=None, lowest_elbo=None, normal=True):
        """An interface for saving a network.
Users may define a method for each subclass for adding a new save-time feature.
Each method should call the _save() method of the superclass in turn.
Users are not expected to call this method directly. Call save() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""


        for i, net in enumerate(self.nets):
            
            if normal:
                net.save_weights(path+"/net"+str(i)+"-"+str(epoch)+".h5")
            else:
                net.save_weights(path+"/net"+str(i)+"-"+str(epoch)+"-"+str(lowest_elbo)+".h5")

        with open(path+"/"+"aux.json", "w") as f:
            json.dump({"parameters":self.parameters,
                       "class"     :self.__class__.__name__,
                       "input_shape":self.net.input_shape[1:]}, f , skipkeys=True, cls=NpEncoder, indent=2)






    def save(self,path="", epoch=None, lowest_elbo=None, normal=True):
       
        
        """An interface for saving a network.
Users should not overload this method; Define _save() for each subclass instead.
This function calls _save bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        #print("Saving the network to {}".format(self.local(path)))
        #os.makedirs(self.local(path),exist_ok=True)

        self._save(path=path, epoch=epoch, lowest_elbo=lowest_elbo, normal=normal)
        print("Network saved")
        return self


    def save_epoch(self, freq=10, path=""):
        def fn(epoch, logs):
            if (epoch % freq) == 0:
                self.save(os.path.join(path,str(epoch)))
        return fn

    def load(self,allow_failure=False,path=""):
        """An interface for loading a network.
Users should not overload this method; Define _load() for each subclass instead.
This function calls _load bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.loaded:
            # print("Avoided loading {} twice.".format(self))
            return

        if allow_failure:
            try:
                print("Loading networks from {} (with failure allowed)".format(self.local(path)))
                self._load(path)
                self.loaded = True
                print("Network loaded")
            except Exception as e:
                print("Exception {} during load(), ignored.".format(e))
        else:
            print("Loading networks from {} (with failure not allowed)".format(self.local(path)))
            self._load(path)
            self.loaded = True
            print("Network loaded")
        return self

    def _load(self,path=""):
        """An interface for loading a network.
Users may define a method for each subclass for adding a new load-time feature.
Each method should call the _load() method of the superclass in turn.
Users are not expected to call this method directly. Call load() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        with open(self.local(os.path.join(path,"aux.json")), "r") as f:

            data = json.load(f)
            _params = self.parameters
            self.parameters = data["parameters"]
            self.parameters.update(_params)
            self.build(tuple(data["input_shape"]))
            self.build_aux(tuple(data["input_shape"]))
        for i, net in enumerate(self.nets):
            # print("JJJJJJJJJJJJJJ")
            # print(self.local(os.path.join(path,f"net{i}.h5")))
            # exit()
            net.load_weights(self.local(os.path.join(path,f"net{i}.h5")))
        
    def reload_with_shape(self,input_shape,path=""):
        """Rebuild the network for a shape that is different from the training time, then load the weights."""
        print(f"rebuilding the network with a new shape {input_shape}.")
        # self.build / self.loaded flag are ignored
        self._build_around(input_shape)
        self._build_aux_around(input_shape)
        for i, net in enumerate(self.nets):
            net.load_weights(self.local(os.path.join(path,f"net{i}.h5")))

    def initialize_bar(self):
        import progressbar
        widgets = [
            progressbar.Timer(format="%(elapsed)s"),
            " ", progressbar.Counter(), " | ",
            # progressbar.Bar(),
            progressbar.AbsoluteETA(format="%(eta)s"), " ",
            DynamicMessage("status", format='{formatted_value}')
        ]
        if "start_epoch" in self.parameters:
            start_epoch = self.parameters["start_epoch"] # for resuming
        else:
            start_epoch = 0
        self.bar = progressbar.ProgressBar(max_value=start_epoch+self.parameters["epoch"], widgets=widgets)

    def bar_update(self, epoch, logs):
        "Used for updating the progress bar."

        if not hasattr(self,"bar"):
            self.initialize_bar()
            from functools import partial
            #self.style = partial(color, fg="black", bg="white")

        tlogs = {}
        for k in self.custom_log_functions:
            tlogs[k] = self.custom_log_functions[k]()
        for k in logs:
            if k[:2] == "t_":
                tlogs[k[2:]] = logs[k]
        vlogs = {}
        for k in self.custom_log_functions:
            vlogs[k] = self.custom_log_functions[k]()
        for k in logs:
            if k[:2] == "v_":
                vlogs[k[2:]] = logs[k]


        #if (epoch % 10) == 9:
        self.bar.update(epoch+1, status = "[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())]) + "\n")
        

        # if (epoch % 10) == 9:
        #     self.bar.update(epoch+1, status = "[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())]) + "\n")
        
        #     f = open(self.path+"/val_metrics.txt", "a")
        #     f.write( "[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())]) )
        #     f.write("\n")
        #     f.close()

        # else:
        #     self.bar.update(epoch+1, status = "[t] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(tlogs.items())]))

    @property
    def net(self):
        return self.nets[0]

    @net.setter
    def net(self,net):
        self.nets[0] = net
        return net

    @property
    def loss(self):
        return self.losses[0]

    @loss.setter
    def loss(self,loss):
        self.losses[0] = loss
        return loss

    def add_metric(self, name,loss=None):
        if loss is None:
            loss = eval(name)
        thunk = lambda *args: loss
        thunk.__name__ = name
        self.metrics.append(thunk)

    def train(self,train_data,
              val_data      = None,
              train_data_to = None,
              val_data_to   = None,
              resume        = False,
              **kwargs):
        """Main method for training.
 This method may be overloaded by the subclass into a specific training method, e.g. GAN training."""


        lowest_elbo = 9999999999.
        i__for_best_files = None

        the_exp_path = self.parameters["the_exp_path"]

        if resume:
            print("resuming the training")
            self.load(allow_failure=False, path="logs/"+self.parameters["resume_from"])
        else:
            #input_shape = train_data.shape[1:]
            #input_shape = (2, 57, 158, 3)
            #input_shape = (2, 25, 70, 3)
            print("type val_data")
            print(type(val_data))
            print(val_data.shape)
            print(val_data.shape[1:])

            if self.parameters["type"] == "vanilla":
                input_shape = val_data.shape[1:]
                #input_shape = 
                image_shape = input_shape[1:]
            else:
                input_shape = (2, 43, 43, 3)
                image_shape = input_shape[1:]

            self.build(input_shape)
            self.build_aux(input_shape)

        epoch      = self.parameters["epoch"]
        batch_size = self.parameters["batch_size"]
        optimizer  = self.parameters["optimizer"]
        lr         = self.parameters["lr"]
        clipnorm   = self.parameters["clipnorm"]
        # clipvalue  = self.parameters["clipvalue"]
        if "start_epoch" in self.parameters:
            start_epoch = self.parameters["start_epoch"] # for resuming
        else:
            start_epoch = 0

        print("train_data shap")
        print(train_data.shape)
        # batch size should be smaller / eq to the length of train_data
        if self.parameters["type"] == "vanilla":
            batch_size = min(batch_size, train_data.shape[0])
        else:
            batch_size = min(batch_size, len(train_data))
        #batch_size = 10


        def make_optimizer(net):
            return getattr(keras.optimizers,optimizer)(
                lr,
                clipnorm=clipnorm
                # clipvalue=clipvalue,
            )

        self.optimizers = list(map(make_optimizer, self.nets))
        self.compile(self.optimizers)

        if val_data     is None:
            val_data     = train_data
        if train_data_to is None:
            train_data_to = train_data
        if val_data_to  is None:
            val_data_to  = val_data

        def replicate(thing):
            if isinstance(thing, tuple):
                thing = list(thing)
            if isinstance(thing, list):
                #assert len(thing) == len(self.nets)
                return thing
            else:
                return [thing for _ in self.nets]

        train_data    = replicate(train_data)
        train_data_to = replicate(train_data_to)
        val_data      = replicate(val_data)
        val_data_to   = replicate(val_data_to)

        plot_val_data = np.copy(val_data[0][:1])
        self.callbacks.append(
            keras.callbacks.LambdaCallback(
                # on_epoch_end = lambda epoch,logs: \
                #     self.plot_transitions(
                #         plot_val_data,
                #         self.path+"/",
                #         epoch=epoch),
                on_train_end = lambda _: self.file_writer.close()))

        def assert_length(data):
            l = None
            for subdata in data:
                if not ((l is None) or (len(subdata) == l)):
                    return False
                l = len(subdata)
            return True

        assert assert_length(train_data   )
        assert assert_length(train_data_to)
        assert assert_length(val_data    )
        assert assert_length(val_data_to )

        def make_batch(subdata):
            # len: 15, batch: 5 -> 3
            # len: 14, batch: 5 -> 2
            # len: 16, batch: 5 -> 3
            # the tail is discarded
            for i in range(len(subdata)//batch_size):
                yield subdata[i*batch_size:(i+1)*batch_size]

        if self.parameters["type"] == "vanilla":
            index_array = np.arange(len(train_data[0]))
        else:
            index_array = np.arange(len(train_data))


        clist = CallbackList(callbacks=self.callbacks)
        clist.set_model(self.nets[0])
        clist.set_params({
            "batch_size": batch_size,
            "epochs": start_epoch+epoch,
            "steps": None,
            "samples": len(train_data[0]),
            "verbose": 0,
            "do_validation": False,
            "metrics": [],
        })
        self.nets[0].stop_training = False




        def unnormalize_colors(normalized_images, mean, std): 
            return (normalized_images*std)+mean


        def deenhance(enhanced_image):
            temp_image = enhanced_image - 0.5
            temp_image = temp_image / 3
            original_image = temp_image + 0.5
            return original_image

        def denormalize(normalized_image, original_min, original_max):
            if original_max == original_min:
                return normalized_image + original_min
            else:
                return (normalized_image * (original_max - original_min)) + original_min





        def generate_logs_vanilla(data,data_to):
            losses = []
            logs   = {}
            for i, (net, subdata, subdata_to) in enumerate(zip(self.nets, data, data_to)):
                evals = net.evaluate(subdata,
                                     subdata_to,
                                     batch_size=batch_size,
                                     verbose=0)
                logs_net = { k:v for k,v in zip(net.metrics_names, ensure_list(evals)) }
                losses.append(logs_net["loss"])
                logs.update(logs_net)
            if len(losses) > 2:
                for i, loss in enumerate(losses):
                    logs["loss"+str(i)] = loss
            logs["loss"] = np.sum(losses)
            return logs

        def generate_logs(data, data_to, epoch=0, forwandb=False):

            images_array = []
            
            for d in data:
                #images_array.append(np.expand_dims(d[0], axis=-1))
                images_array.append(d[0])
            images_array = np.array(images_array)


            actions_array = []

            for d in data:
                actions_array.append(d[1])

            actions_array = np.array(actions_array)

            losses = []
            logs   = {}
            
            # print(images_array.shape) # (4500, 2, 48, 48, 1)
            # print(actions_array.shape) # (4500, 24)
            
            # NEW VERSION
            evals = self.nets[0].evaluate([images_array, actions_array],
                                            images_array,
                                            batch_size=batch_size,
                                            verbose=0)


            preds = self.nets[0].predict([images_array, actions_array])
            print("PREDS") # (600, 2, 48, 48, 1)
            # (1, 2, 25, 70, 3)

            print(preds.shape)

            if epoch % 2 and epoch > 0:
                iiii = 0
                #iiii = random.randint(0, 1) # 
                #plot_image(np.reshape(preds[iiii,1,:,:,:].squeeze(), (30, 45, 3)),"THEPREDICTION-BLOCKSWORLD.png")
                #plot_image(np.reshape(preds[iiii,1,:,:,:].squeeze(), (48, 48)),"THEPREDICTION-PUZZ.png")
                # (4, 16, 3) HANOI
                #plot_image(np.reshape(preds[iiii,1,:,:,:].squeeze(), (4, 16, 3)),"THEPREDICTION-HANOI.png")
                #theimage = np.reshape(preds[iiii,1,:,:,:].squeeze(), (56, 120, 3))
                # 28, 60, 3.

                theimage = np.reshape(preds[iiii,1,:,:,:].squeeze(), image_shape)

                theimage = unnormalize_colors(np.squeeze(theimage), self.parameters["mean"], self.parameters["std"])
                

                theimage = deenhance(theimage)

                theimage = denormalize(theimage, self.parameters["orig_min"], self.parameters["orig_max"])
                
                theimage = np.clip(theimage, 0, 1)
                
                plt.imsave("THEPREDICTION-NoisyPartialDFA2.png", theimage)

            logs_net = { k:v for k,v in zip(self.nets[0].metrics_names, ensure_list(evals)) }
        
            #if forwandb:
            to_send_towandb = logs_net
            to_send_towandb["epoch"] = epoch
            #wandb.log(to_send_towandb)

            losses.append(logs_net["loss"])
            logs.update(logs_net)
            if len(losses) > 2:
                for i, loss in enumerate(losses):
                    logs["loss"+str(i)] = loss
            logs["loss"] = np.sum(losses)
            return logs


        if self.parameters["type"] == "vanilla":

            try:

                clist.on_train_begin()
                logs = {}

                for epoch in range(start_epoch,start_epoch+epoch):
                    np.random.shuffle(index_array)
                    indices_cache       = [ indices for indices in make_batch(index_array) ]
                    train_data_cache    = [[ train_subdata   [indices] for train_subdata    in train_data    ] for indices in indices_cache ]
                    train_data_to_cache = [[ train_subdata_to[indices] for train_subdata_to in train_data_to ] for indices in indices_cache ]
                    clist.on_epoch_begin(epoch,logs)
                    for train_subdata_cache,train_subdata_to_cache in zip(train_data_cache,train_data_to_cache):
                        for net,train_subdata_batch_cache,train_subdata_to_batch_cache in zip(self.nets, train_subdata_cache,train_subdata_to_cache):
                            
                            
                            net.train_on_batch(train_subdata_batch_cache, train_subdata_to_batch_cache)


                            # logs = {}

                            # for k,v in generate_logs(val_data,  val_data_to, epoch=epoch, batch=some_, forwandb=self.parameters["use_wandb"]).items():
                            #     logs["v_"+k] = v
                            # clist.on_batch_end(epoch,logs)

                    logs = {}

                    if epoch > 0 and epoch % 250 == 0:
                        print("lowest_elbolowest_elbolowest_elbolowest_elbo")
                        print(lowest_elbo)

                        self.save(path = the_exp_path, epoch=epoch, lowest_elbo="None", normal=True)


                    for k,v in generate_logs_vanilla(train_data, train_data_to).items():
                        logs["t_"+k] = v
                    for k,v in generate_logs_vanilla(val_data,  val_data_to).items():
                        logs["v_"+k] = v
                    clist.on_epoch_end(epoch,logs)
                    if self.nets[0].stop_training:
                        break
                clist.on_train_end()

            except KeyboardInterrupt:
                print("learning stopped\n")
            finally:
                #self.save()
                self.loaded = True
            return self

        else:

            try:

                clist.on_train_begin()
                logs = {}

                for epoch in range(start_epoch, start_epoch+epoch):
                    np.random.shuffle(index_array)

                    indices_cache       = [ indices for indices in make_batch(index_array) ]
                    train_data_cache = [[train_data[i] for i in indices_cache[j]] for j in range(len(indices_cache))]
                    train_data_to_cache = [[ train_data_to[i] for i in indices_cache[j]] for j in range(len(indices_cache))]

                    clist.on_epoch_begin(epoch,logs)
                    for train_subdata_cache,train_subdata_to_cache in zip(train_data_cache,train_data_to_cache):
                        #for net,train_subdata_batch_cache,train_subdata_to_batch_cache in zip(self.nets, train_subdata_cache,train_subdata_to_cache):
                        #    #net.train_on_batch(train_subdata_batch_cache, train_subdata_to_batch_cache)

                        net = self.nets[0]

                        #self.parameters["present_xys"] = self.parameters["x_and_ys"][batch_count]

                        # print("SHAPE SUBDATA")
                        # print(len(train_subdata_cache))
                        # # 

                        # images (each item[0] of train_subdata_cache)
                        #x_data = np.array([np.expand_dims(item[0], axis=-1) for item in train_subdata_cache])
                        x_data = np.array([item[0] for item in train_subdata_cache])


                        # actions (each item[1] of train_subdata_cache)
                        action_input_data = []
                        for item in train_subdata_cache:

                            action_input_data.append(item[1])

                        action_input_data = np.array(action_input_data)
    
                        net.train_on_batch([x_data, action_input_data], x_data)
                    
                    
                    logs = {}

                    for k,v in generate_logs(val_data,  val_data_to, epoch=epoch, forwandb=self.parameters["use_wandb"]).items():
                        logs["v_"+k] = v
                        if k == "elbo":
                            if float(v) < lowest_elbo:
                                lowest_elbo = float(v)
                                i_for_best_files = epoch
                    clist.on_epoch_end(epoch,logs)
                    if self.nets[0].stop_training:
                        break

                    if epoch > 0 and epoch % 250 == 0:
                        print("lowest_elbolowest_elbolowest_elbolowest_elbo")
                        print(lowest_elbo)

                        self.save(path = the_exp_path, epoch=epoch, lowest_elbo=str(int(lowest_elbo)), normal=False)

                #wandb.finish()
                clist.on_train_end()


            except KeyboardInterrupt:
                print("learning stopped\n")
            finally:
                self.save(epoch=epoch)
                self.loaded = True
            return self

    def evaluate(self,*args,**kwargs):

        return np.sum([
            { k:v for k,v in zip(net.metrics_names,
                                 ensure_list(net.evaluate(*args,**kwargs)))}["loss"]
            for net in self.nets
        ])

    def report(self,train_data,
               batch_size=1000,
               test_data=None,
               train_data_to=None,
               test_data_to=None):
        pass

    def save_array(self,name,data):
        print("Saving to",self.local(name))
        with open(self.local(name), "wb") as f:
            np.savetxt(f,data,"%d")


    def _plot(self,path,columns,epoch=None):
        """yet another convenient function. This one swaps the rows and columns"""
        columns = list(columns)
        if epoch is None:
            rows = []
            for seq in zip(*columns):
                rows.extend(seq)
            from .util.plot import plot_grid
            plot_grid(rows, w=len(columns), path=path, verbose=True)
            return
        else:
            # assume plotting to tensorboard

            if (epoch % 10) != 0:
                return

            _, basename = os.path.split(path)
            import tensorflow as tf
            import keras.backend.tensorflow_backend as K
            import imageio
            import skimage

            # note: [B,H,W] monochrome image is handled by imageio.
            # usually [B,H,W,1].
            for i, col in enumerate(columns):
                col = skimage.util.img_as_ubyte(np.clip(col,0.0,1.0))
                for k, image in enumerate(col):
                    self.file_writer.add_summary(
                        tf.Summary(
                            value = [
                                tf.Summary.Value(
                                    tag  = basename.lstrip("_")+"/"+str(i),
                                    image= tf.Summary.Image(
                                        encoded_image_string =
                                        imageio.imwrite(
                                            imageio.RETURN_BYTES,
                                            image,
                                            format="png")))]),
                        epoch)
            return
