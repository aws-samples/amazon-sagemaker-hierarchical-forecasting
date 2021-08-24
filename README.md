## Hierarchical Forecasting using Amazon SageMaker
----
Time series forecasting is a very common and well known problem in machine learning and statistics. Most of the times, the time series data follows a hierarchical aggregation structure.  For e.g. in retail, weekly sales for a SKU at a store can roll up to different geographical hierarchies at city, state or country level. In these cases we need to ensure, that the sales estimates are in agreement, when rolled up to a higher level. In such scenarios, *Hierarchical Time Series Forecasting*, which is the process of* *generating coherent forecasts* *(or reconciling* *incoherent forecasts),* *allowing individual time series to be forecast individually, but preserving the relationships within the hierarchy, is used.
Many customers are either using hierarchical forecasting methods or have an upcoming use case that requires hierarchical forecasting to achieve better results. In this notebook we take the example of demand forecasting on synthetic retail data and show you how to train and tune multiple hierarchichal time series models across algorithms and hyper-parameter combinations using the [`scikit-hts`](#https://scikit-hts.readthedocs.io/_/downloads/en/stable/pdf/) toolkit on Amazon SageMaker. We will first show you how to setup scikit-hts on SageMaker using the SKLearn estimator, then train multiple models using SageMaker Experiments, and finally use SageMaker Debugger to monitor suboptimal training and improve training efficiencies. We will walk you through the following steps:

1.	Setup
2.	Prepare Time Series Data
    - Data Visualization
    - Split data into train and test
    - Hierarchical Representation
    - Visualizing the tree structure
3.	Setup the scikit-hts training script
4.  Setup Amazon SageMaker Experiment and Trials
5.	Setup the SKLearn Estimator
6.	Evaluate metrics and select a winning candidate
7.	Run time series forecasts
    - Visualization at Region Level
    - Visualization at State Level

## Amazon SageMaker
----
Amazon SageMaker is the most comprehensive and full managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers. It also provides common machine learning algorithms that are optimized to run efficiently against extremely large data in a distributed environment. With native support for bring-your-own-algorithms and frameworks, SageMaker offers flexible distributed training options that adjust to your specific workflows. Deploy a model into a secure and scalable environment by launching it with a few clicks from SageMaker Studio or the SageMaker console.  We use Amazon SageMaker Studio for running the code, for more details see the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html).

## How to run the code in Amazon SageMaker Studio? 
----
If you haven't used Amazon SageMaker Studio before, please follow the steps mentioned in [`Onboard to Amazon SageMaker Studio`](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html).

### To log in from the SageMaker console

- Onboard to Amazon SageMaker Studio. If you've already onboarded, skip to the next step.
- Open the SageMaker console.
- Choose Amazon SageMaker Studio.
- The Amazon SageMaker Studio Control Panel opens.
- In the Amazon SageMaker Studio Control Panel, you'll see a list of user names.
- Next to your user name, choose Open Studio.

### Open a Studio notebook
SageMaker Studio can only open notebooks listed in the Studio file browser. In this example we will `Clone a Git Repository in SageMaker Studio`.

#### To clone the repo

- In the left sidebar, choose the File Browser icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/File_browser_squid.png'> ).
- Choose the root folder or the folder you want to clone the repo into.
- In the left sidebar, choose the Git icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/Git_squid.png'>  ).
- Choose Clone a Repository.
- Enter the URI for the repo https://github.com/aws-samples/amazon-sagemaker-hierarchical-forecasting.git.
- Choose CLONE.
- If the repo requires credentials, you are prompted to enter your username and password.
- Wait for the download to finish. After the repo has been cloned, the File Browser opens to display the cloned repo.
- Double click the repo to open it.
- Choose the Git icon to view the Git user interface which now tracks the examples repo.
- To track a different repo, open the repo in the file browser and then choose the Git icon.

### To open a notebook

- In the left sidebar, choose the File Browser icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/File_browser_squid.png'> ) to display the file browser.
- Browse to a notebook file and double-click it to open the notebook in a new tab.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

