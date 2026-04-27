def split_prices(prices, train_ratio=0.7):
    """
    Split price data into training and unseen testing data.

    The agent trains only on train_prices.
    Then we evaluate it on test_prices to check generalization.
    """

    split_index = int(len(prices) * train_ratio)

    train_prices = prices[:split_index]
    test_prices = prices[split_index:]

    return train_prices, test_prices