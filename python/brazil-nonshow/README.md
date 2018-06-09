# ml-brazil-noshow

This dataset captures information about the medical appointment bookings in Brazil. A total of 100k records are provided in the dataset which contains details about the patient, neighbourhood, the time at which the appointment was scheduled, the date on which the appointment, several factors and finally whether the patient showed up for the appointment or not.This dataset is sourced from [kaggle](https://www.kaggle.com/joniarroba/noshowappointments)



By analaysing this dataset, we hope to answer several questions as to why the patients do not turn up for their appointments. These may include

* How does the neighbourhood affect the no show percentage?
* Are older patients likely to honour their appointments more than the younger ones?
* Do patients with specific diseases show up for appointments more often?
* Does the number of days between the shceduled date and appointment date have a bearing on the no show?
* Do innovative services like a SMS reminder help in reducing no shows?

## Prerequisites

Create a clone of anoconda environment using below command : 
* conda env create -f environment.yml

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
