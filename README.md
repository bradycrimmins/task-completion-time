# task-completion-time
Consider the following scenario in which we want to use Python to preprocess data queried from a series of tables on a SQL Server database populated by a warehouse management software system (WMS). The end goal is to engineer a set of features that can be fed to scikit-learn and/or TensorFlow for training a linear regression ML model against a label of Task Completion Time (defined further below). 

Tasks in the WMS are defined as individual packages of actionable work that can be assigned to warehouse employees for execution, and they span different task types like Receiving, Putaway, Picking, Packing, Replenishment, Cycle Counting, Inventory Control, etc. 

Tasks are comprised of a header table and child table as defined further below. The detail records on the child table are all created against a common task ID from the parent table, and these child records ultimately define the sequence in which a user who has been assigned the given task must travel a certain distance from location to location within the warehouse and at each stop perform a certain action on a specified quantity of units of a particular item using a specified piece of equipment like a forklift, a picking tote, a picking cart, a swing-reach machine, etc. Once all these details/stops on the task are completed, the header record is updated to a completed status, and a completion timestamp is applied. There will be a delta between this completion timestamp and a separate column for the creation timestamp when the task was originally created.

Ultimately, our label of interest for the ML model is precisely this delta or the calculated difference between the completion time and creation time of a given task. Our features will be engineered against the label using relevant datapoints from a series of tables. We are interested in using linear regression to determine whether there is a particular mix of features that drives a statistically significant difference in this total time required to complete a given task record, so that we can eventually create an equation for predicting completion times for future open tasks and feed this information in real time to department supervisors and warehouse managers so they can make staffing adjustments on the fly or otherwise implement new policies as far as how work is released and assigned to the warehouse floor. 

For this exercise assume we have data from the past three years of WMS operations and all the task records in our dataset are already completed. 

Further assumptions are included towards the end of this text. 

To begin, consider the following list of relevant tables, with a description of the data therein:

•	LOCN_HDR – contains a catalog of location records as points in the warehouse represented as a simple grid; users must travel to these locations to complete task details  
  o	Subset of relevant columns:
    	LOCN_ID (primary key) – unique 6-digit integer next-up value representing an individual location in the warehouse 
    	X_COORDINATE – integer value from 0 to 99999 denoting the position of a location from left to right relative to the warehouse’s main wall, used as an x-axis, where a value of 1 = 1 ft, 2 = ft, 3 = 3ft …   99999 = 99999 ft 
    	Y_COORDINATE – integer value from 0 to 99999 denoting the position of a location from left to right relative to the adjacent wall in the warehouse, used as a y-axis, where a value of 1 = 1 ft, 2 = ft, 3 = 3ft … 99999 = 99999 ft 
•	ITEMS – contains a catalog of items available in the warehouse and allocated on task details
  o	Subset of relevant columns:
    	ITEM_ID (primary key) – a unique 10-digit integer next-up value denoting an individual item
    	PRODUCT_CLASS – a 
    	WEIGHT – a decimal value denoting the kg weight of the item rounded to two decimal places 
    	VOLUME – a decimal value denoting the cubic inch volume of the item rounded to two decimal places 
•	USERS – contains a catalog of warehouse employees to whom tasks are assigned
  o	Subset of relevant columns:
    	USER_ID (primary key) – a unique 6-digit integer value denoting an individual employee
    	DEPARTMENT_CODE – an integer code defining the warehouse department to which a user belongs 
    	SHIFT_ID – an integer code defining the shift to which a user belongs (1 for First Shift, 2 for Second Shift, 3 for Third Shift) 
    	HIRE_DATE – datetime column denoting when the user was hired 
•	TASK_HDR – contains header-level information for the task records.
  o	Subset of relevant columns: 
    	TASK_ID (primary key) – a unique 8-digit integer next-up value denoting one individual task
    	TASK_TYPE – a varchar column denoting the type of task from possible categories Receiving, Putaway, Picking, Packing, Replenishment, Cycle Counting, Inventory Control
    	USER_ID (foreign key onto USERS.USER_ID) 
    	EQUIPMENT_ID – integer from 0 to 4 specifying the equipment type required for completing the task 
    	CREATED_DATE_TIME – datetime column representing the moment the task was created
    	COMPLETED_DATE_TIME – datetime column representing the moment the task was completed
•	TASK_DTL – contains detail-level information for the task records 
  o	Subset of relevant columns:
    	TASK_DTL_ID (primary key) – a 12-digit integer next-up value denoting one individual task detail, one individual stop on the task 
    	TASK_ID (foreign key onto TASK_HDR.TASK_ID)
    	SEQ_NBR – integer from 0 to 99 specifying the order in which task details are to be completed (note that these ranges do not repeat across the same TASK_ID but can be recycled on others)  
    	PULL_LOCN_ID (foreign key onto LOCN_HDR.LOCN_ID) – note that this range doesn’t repeat across the same TASK_ID but can be repeated on others, i.e. you can’t visit the same location twice on the same task 
    	ITEM_ID (foreign key onto ITEMS.ITEM_ID)
    	QTY_PULLD – integer from 0 to 99 specifying the number of units of an item required on a given task detail 

Here is the proposed feature list to train our model, with definitions:
•	Task Type = the TASK_HDR.TASK_TYPE value
•	Travel Distance = the total distance travelled in feet on a given task as defined by the summation of all the individual distances from one TASK_DTL.PULL_LOCN_ID to the next, utilizing the x and y coordinates on the matching LOCN_HDR record
•	User Experience Level = the datetime difference between USERS.HIRE_DATE and the current date, expressed as an integer number of months 
•	User Department = the USER.DEPARTMENT_CODE value 
•	User Shift = the USER.SHIFT_CODE value 
•	Day of Week = the descriptive weekday name from TASK_HDR.CREATED_DATE_TIME 
•	Hour of Day = the hour value from TASK_HDR.CREATED_DATE_TIME
•	Month = the month value from TASK_HDR.COMPLETED_DATE_TIME
•	Equipment = the TASK_HDR.EQUIPMENT_ID value
•	Total Quantity = the summation of the TASK_DTL.QTY_PULLD on one TASK_DTL.TASK_ID 
•	Total Weight = the summation of TASK_DTL.QTY_PULLD * ITEMS.WEIGHT for the matching TASK_DTL.ITEM_ID record, for all records against the same TASK_DTL.TASK_ID
•	Total Volume = the summation of TASK_DTL.QTY_PULLD * ITEMS.WEIGHT for the matching TASK_DTL.ITEM_ID record, for all records against the same TASK_DTL.TASK_ID
•	Distinct Items = a distinct count of TASK_DTL.ITEM_ID for one TASK_DTL.TASK_ID 
•	Distinct Product Classes = a distinct count of ITEMS.PRODUCT_CLASS values associated with a particular TASK_DTL.TASK_ID 

Here is our label definition:

Completion Time = the datetime difference between TASK_HDR.COMPLETED_DATE_TIME – TASK_HDR.CREATED_DATE_TIME 

Other assumptions:
•	There are 50000 locations in the warehouse and their x and y coordinate positions on the grid are randomized but no combination of x and y is repeated 
•	TASK_HDR.COMPLETED_DATE_TIME > TASK_HDR.CREATED_DATE_TIME for all records, but the COMPLETED_DATE_TIME is at least always on the same day as the CREATED_DATE_TIME 
