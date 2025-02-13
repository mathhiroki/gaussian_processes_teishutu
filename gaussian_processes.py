import numpy as np
import typing as T
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Class definitions


##########################################
# Kernel Class
##########################################


class BaseKernels(object):
    """
    Base kernel class
    """

    def __init__(self, name: str, params: dict) -> None:
        """
        Initialization of base kernel object

        Args:
            name (str): type of the kernel
            params (dict): parameters
        """
        self._name = name
        self._params = params

        return

    @property
    def name(self) -> str:
        """
        The name (type) of this kernel
        """
        return self._name

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute the kernel of X and Y

        Args:
            X (np.ndarray): input x
            Y (np.ndarray): input y

        Returns:
            kernel (np.ndarray): output of kernel
        """
        raise NotImplementedError


class RBF(BaseKernels):
    """
    RBF kernel class
    """

    def __init__(self, name: str, params: dict) -> None:
        """
        Initialization of base kernel object

        Args:
            name (str): type of the kernel
            params (dict): parameters
        """
        super().__init__(name, params)

        ##########################################
        self.sigma = params.get("sigma", 1)
        ##########################################

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute the RBF (Gaussian) kernel of X and Y

        K(X, Y) = (1/sqrt(2pi)*sigma)^d * exp(-||X-Y||^2/(2sigma^2))

        Args:
            X (np.ndarray): input x
            Y (np.ndarray): input y

        Returns:
            kernel (np.ndarray): output of kernel
        """
        assert X.shape[1] == Y.shape[1], "Dimension mismatch."

        ##########################################
        d = X.shape[1]
        K = ((1 / (np.sqrt(2 * np.pi) * self.sigma)) ** d)* np.exp(-(np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)) / (2 * self.sigma**2))
        ##########################################


        return K


class KernelFactory(object):
    """
    Spawning kernels
    """

    def __new__(cls, name: str, params: dict) -> T.Any:
        """
        Spawning renderer

        Args:
            name (str): name of the kernels
            params (dict): dictionary of parameters

        Returns:
            kernels (BaseKernels): kernel object
        """
        if name == "rbf":
            ##########################################
            kernels = RBF(name,params)
            ##########################################

        else:
            ##########################################
            raise NotImplementedError
            ##########################################


        return kernels


##########################################
# Regressor Class
##########################################


class GP(object):
    """
    Gaussian processes class
    """

    def __init__(
        self,
        kernel: BaseKernels,
        n_features: int,
        noise_cov: float,
    ) -> None:
        """
        Initializing GP object

        Args:
            kernel (BaseKernels): kernel functions to use in GP
            n_features (int): number of features (dimension of input)
            noise_cov (float): covariance of noise
        """
        self.kernel = kernel
        self.noise_cov = noise_cov
        self.n_features = n_features
        self.x: T.Optional[np.ndarray] = None
        self.t: T.Optional[np.ndarray] = None
        self.Kinv: T.Optional[np.ndarray] = None

    def update(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Updating the GP model

        Args:
            x (np.ndarray): input of dimension (n_samples, n_features)
            t (np.ndarray): output of dimension (n_samples, 1)
        """
        assert (x.shape[1] == self.n_features) and (
            t.shape[1] == 1
        ), "dimension mismatch!"
        self._update_data(x, t)
        self._update_matrix()

    def _update_data(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        Updating the data point

        Args:
            x (np.ndarray): input of dimension (n_samples, n_features)
            t (np.ndarray): output of dimension (n_samples, 1)
        """
        ##########################################
        if self.x is None: #de-tanasi
            self.x = x
            self.t = t
        else:
            self.x = np.vstack((self.x,x))
            self.t = np.vstack((self.t,t))
        ##########################################

    def _update_matrix(self) -> None:
        """
        Updating the matrices
        """
        assert self.x is not None, "Invalid call."
        ##########################################
        K = self.kernel(self.x,self.x) + self.noise_cov*np.eye(self.x.shape[0])
        self.Kinv = np.linalg.inv(K)
        ##########################################

    def predict(self, x: np.ndarray) -> T.Tuple[float, float]:
        """
        Predinting the value and variance of given point

        Args:
            x (np.ndarray): input of dimension (n_features)

        Returns:
            mean and variance (Tuple[float, float]): predicted mean & var
        """
        assert self.x is not None, "No data observed."
        assert x.shape == (self.n_features,), "input dimension mismatch."
        ##########################################
        k = self.kernel(self.x , x.reshape(1,-1))
        m = k.T @ self.Kinv @ self.t
        v = self.kernel(x.reshape(1,-1),x.reshape(1,-1)) - k.T @ self.Kinv @ k
        ##########################################

        return (m.item(), v.item())


##########################################
# Generator Class
##########################################


class BaseDataGenerator(object):
    """
    Base data generator
    """

    def __init__(self, name: str, params: dict) -> None:
        """
        Initialization of base data generator

        Args:
            name (str): type of the generator
            params (dict): parameters
        """
        self._name = name
        self._params = params

        return

    @property
    def name(self) -> str:
        """
        The name (type) of this generator
        """
        return self._name

    def __iter__(self) -> T.Any:
        """
        Self already has __next__
        """
        return self

    def __next__(self) -> T.Any:
        """
        Return next iterated value

        Return:
            val (T.Any): value
        """
        raise NotImplementedError


class BasicGenerator(BaseDataGenerator):
    """
    Basic generator to use in this class
    """

    def __init__(
        self,
        name: str,
        params: dict,
    ) -> None:
        """
        Initializing the basic generator

        Args:
            name (str): type of the generator
            params (dict): parameters
        """
        super().__init__(name, params)
        assert isinstance(self._params["n_features"], int), "n_features is invalid"
        assert isinstance(self._params["min"], int), "min is invalid"
        assert isinstance(self._params["max"], int), "max is invalid"
        assert isinstance(self._params["data_num"], int), "data_num is invalid"
        assert isinstance(
            self._params["noise_cov"], (int, float)
        ), "noise_cov is invalid"
        ##########################################
        self.x_data = np.random.uniform(self._params["min"], self._params["max"], self._params["data_num"]).reshape(-1, 1)
        self.y_data = self.compute_y(self.x_data) + np.random.normal(0, self._params["noise_cov"], size = self.x_data.shape)
        self.index = 0
        ##########################################

    @staticmethod
    def compute_y(xdata: np.ndarray) -> np.ndarray:
        """
        Computing the output

        Args:
            xdata (np.ndarray): input array
        

        Returns:
            ydata (np.ndarray): output array
        """
        return 2 * np.sin(xdata)
    def __iter__(self):
        self.idx = 0  # イテレーション開始時にリセット
        return self

    def __next__(self) -> T.Any:
        """
        Return next iterated value

        Return:
            val (T.Any): value
        """

        ##########################################
        if self.idx >= len(self.x_data):
            raise StopIteration
        ret_value = (self.x_data[self.idx], self.y_data[self.idx])
        self.idx += 1
        ##########################################
        return ret_value


# Function definitions


def plotdata(gp: GP, plotx: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Plot data of the GP

    Args:
        gp (GP): gp object
        plotx (np.ndarray): input array for plotting

    Returns:
        m_vec (np.ndarray): mean vector
        v_vec (np.ndarray): variance vector
    """
    m_vec = np.zeros_like(plotx)
    v_vec = np.zeros_like(plotx)
    for i, x in enumerate(plotx):
        m_vec[i], v_vec[i] = gp.predict(np.expand_dims(x, axis=0))

    return m_vec, v_vec


def create_movie(
    plotx: np.ndarray,
    g_vec: np.ndarray,
    mean_list: list,
    var_list: list,
    input_list: list,
    output_list: list,
) -> None:
    """
    Creating the movie of plots of GP

    Args:
        plotx (np.ndarray): input vec
        g_vec (np.ndarray): ground truth output of function
        mean_list (list): list of mean vectors
        var_list (list): list of variacne vectors
        input_list (list): list of inputs
        output_list (list): list of outputs
    """
    fig, ax = plt.subplots(figsize=(13, 9))

    def updater(frame: int) -> T.Any:
        ax.cla()
        ax.grid()
        ax.plot(plotx, g_vec, linewidth=4.0, color="black")
        ax.scatter(input_list[: frame + 1], output_list[: frame + 1], s=100, c="red")
        ax.plot(plotx, mean_list[frame], linewidth=4.0, color="blue")
        ax.fill_between(
            plotx,
            mean_list[frame] - var_list[frame],
            mean_list[frame] + var_list[frame],
            alpha=0.3,
            facecolor="blue",
        )
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)

    ani = animation.FuncAnimation(
        fig=fig, func=updater, frames=len(mean_list), interval=1000
    )
    # ani.save("movie.mp4", writer="ffmpeg")
    ani.save(r"path_save_MP4", writer="ffmpeg")

    plt.close()


def main():
    """
    Main function
    """
    data_params = {"n_features": 1, "min": -5, "max": 5, "data_num": 50, "noise_cov": 1}
    datagen = BasicGenerator("basic", data_params)

    kernelparams = {"sigma": 0.2}
    kernel = KernelFactory("rbf", kernelparams)

    gp = GP(kernel, 1, 1)

    # Save data for later use (better to use data object)
    mean_list = []
    var_list = []
    input_list = []
    output_list = []
    plt_min, plt_max, plt_num = -5, 5, 100
    plotx = np.linspace(plt_min, plt_max, plt_num)
    g_vec = datagen.compute_y(plotx)

    for data in datagen:
        input_list.append(data[0])
        output_list.append(data[1])
        x = np.expand_dims(data[0], axis=0)
        t = np.expand_dims(data[1], axis=0)
        gp.update(x, t)
        m, v = plotdata(gp, plotx)
        mean_list.append(m)
        var_list.append(v)

    create_movie(plotx, g_vec, mean_list, var_list, input_list, output_list)


##########################################
main()
##########################################
