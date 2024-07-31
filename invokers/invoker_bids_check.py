# from bids_validator import BIDSValidator

# # Initialize the validator
# validator = BIDSValidator()

# # Path to the dataset folder
# dataset_path = '/imaging/projects/cbu/kymata/data/for-export-as-BIDS'

# import ipdb;ipdb.set_trace()

# # Run the validation
# validation_report = validator.validate(dataset_path)

# # Print or process the validation report
# print(validation_report)

from bids_validator import validate
validate.BIDS('/imaging/projects/cbu/kymata/data/for-export-as-BIDS')