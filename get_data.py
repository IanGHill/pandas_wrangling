import requests


def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    # Raise an error if the request was unsuccessful
    response.raise_for_status()

    # Open a local file with the same name as the downloaded file
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"{filename} has been downloaded.")


# Example usage
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
filename = "usedcars.csv"
download_file(url, filename)
