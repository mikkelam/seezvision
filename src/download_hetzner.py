import asyncio
import csv
import os
import traceback

from asyncio import ensure_future, CancelledError
from pathlib import Path

import aiofiles
import aiohttp
import numpy as np
import pandas as pd
from aiohttp import BasicAuth
from timeit import default_timer as timer
import logging

from concurrent.futures import TimeoutError
from aiohttp.client_exceptions import ServerDisconnectedError, ClientConnectorError
from tenacity import retry, wait_exponential

logging.basicConfig(level=logging.DEBUG)

data_dir = Path(__file__).parent / 'data'
# data_dir = Path('/Volumes/shared/data')
# data_dir = Path('/Users/mikkelam/data')
data_dir.mkdir(exist_ok=True)

base_url = 'https://u213553-sub6.your-storagebox.de'


class AsyncEnumerate:
    """Asynchronously enumerate an async iterator from a given start value"""

    def __init__(self, asequence, start=0):
        self._asequence = asequence
        self._value = start

    async def __anext__(self):
        elem = await self._asequence.__anext__()
        value, self._value = self._value, self._value + 1
        return value, elem

    def __aiter__(self):
        return self


def get_paths(ad_id, p_hash):
    remote_path = Path('/images') / ad_id[0:2] / ad_id[2:4] / ad_id[4:] / (p_hash + '_thumb.jpg')
    local_path = data_dir / ad_id / f'{p_hash}.jpg'
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return False, False
    url = f'{base_url}{str(remote_path)}'
    return url, str(local_path)


# start = timer()

sem = asyncio.BoundedSemaphore(140)


@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
async def download_file(session: aiohttp.ClientSession, ad_id, p_hash, idx):
    url, local_path = get_paths(ad_id, p_hash)
    if not local_path:
        return True

    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(local_path, mode='wb')
                    await f.write(await resp.read())
                    await f.close()
                    return True
                else:
                    print(f'failed {idx}')
                    return False
        except CancelledError as e:
            print('weird cancel error')
        except (TimeoutError, aiohttp.ClientConnectorSSLError, ServerDisconnectedError,
                ClientConnectorError) as e:
            pass
        except Exception as e:
            pass

    return False


async def download_multiple(session: aiohttp.ClientSession, rows):
    batch_size = 10000
    batch = []
    batch_total = 1028098 // batch_size
    start = timer()
    async for idx, row in AsyncEnumerate(rows):
        batch_num = idx // batch_size
        # if (batch_num) < 31:
        #     continue
        row = list(csv.reader([row], delimiter=',', quotechar='"'))[0]
        ad_id = row[0]
        p_hashes = row[-1].split(',')
        for idx2, p_hash in enumerate(set(p_hashes)):
            batch.append(
                ensure_future(download_file(session, ad_id, p_hash, idx)))
        if idx % batch_size == 0:
            result = await asyncio.gather(*batch)

            result = np.array(result)
            fails = (1 - result).sum()

            print(
                f'Batch {batch_num}/{batch_total}. '
                f'Fails: {fails}/{batch_size}. '
                f'Time: {(timer() - start):.2f}s')
            start = timer()
            batch = []
    await asyncio.gather(*batch)  # last batch

    return


login = os.environ['HETZNER_LOGIN']
password = os.environ['HETZNER_PASSWORD']


async def main():
    async with aiofiles.open('cars.csv', 'r') as rows:
        config = {'connector': aiohttp.TCPConnector(limit=150),
                  # 'headers': {'Connection': 'close'},
                  'auth': BasicAuth(login=login,
                                    password=password)}
        async with aiohttp.ClientSession(**config) as session:
            await download_multiple(session, rows=rows)
            print('finished')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
# print(len(list(Path('images/').rglob('*.jpg'))))
# print((np.array(results) * 1).sum())
...
