 # Set these variables
EMAIL='YOUR_EMAIL@example.com'
APIKEY='YOUR_API_KEY_HERE'

if [ $# -ne 1 ]
then
    echo "ERROR - File containing JSON array of signature IDs is needed!"
    echo "Usage: $0 <input file>"
    exit 1
fi

FILETYPE="\"filetype\": [\"raw\"], "
TEMPLATE="{\"email\": \"${EMAIL}\", \"apikey\": \"${APIKEY}\", ${FILETYPE} \"output\": \"data\", \"sigid\": __SIGID__}"
sigids=$(head -1 $1 | tr -d '[]' | tr -d ',')
if [ "x${sigids}" = "x" ]
then
    echo "ERROR - JSON array in $1 appears to be invalid!"
    exit 1
fi

for i in $sigids
do
    echo "Getting compressed / zip CSV data for sigid $i ..."
    echo $TEMPLATE | sed 's/__SIGID__/'${i}'/' > request.json
    curl -H "Content-Type: application/json" -o sigid-${i}.zip -d @request.json https://gesl.ornl.gov/api/apps/gesl
done 