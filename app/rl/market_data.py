import csv


def load_prices_from_csv(
    file_path="data/btc.csv",
    price_column="Close",
    limit=500,
):
    prices = []

    previous_price = None

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            price = float(row[price_column])

            if price <= 0:
                continue

            if previous_price is not None and price == previous_price:
                continue

            prices.append(price)
            previous_price = price

            if len(prices) >= limit:
                break

    return prices