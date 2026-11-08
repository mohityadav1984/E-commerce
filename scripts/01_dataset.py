import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import dataset
r = dataset.validate(dataset.ORDERS)
print(f"seed={dataset.SEED} orders={len(dataset.ORDERS)}")
print("category weights:", dict(zip(dataset.CATEGORIES, dataset.CATEGORY_WEIGHTS)))
print("status weights  :", dict(zip(dataset.STATUSES, dataset.STATUS_WEIGHTS)))
print(f"amount range    : INR {dataset.AMOUNT_MIN}-{dataset.AMOUNT_MAX} (log-uniform)")
print("count per category:", r["by_category"])
print("count per status  :", r["by_status"])
print(f"delayed_shipment %: {r['delayed_pct']}")
print("problems:", r["problems"] or "none")
print("first 3 records:")
for o in dataset.ORDERS[:3]:
    print("  ", o)
