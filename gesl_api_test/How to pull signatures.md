## How to pull from GESL using GESL API
1. Create a list of Signature IDs that you want to pull in the form of `sigids.json`. This `.json` file can similarly be created using a cURL request to the GESL API.
2. Run `extract_gesl_signatures.sh` to pull the `.zip` files associated with each signature in `sigids.json`, ensuring that you are using your own email and API key (you can get this from the [website](https://gesl.ornl.gov/application) itself). 
3. Run `extract_zip.py` to unzip all the files.
