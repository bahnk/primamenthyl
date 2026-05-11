{{ config(tags=['individual']) }}

with bins as (
    select generate_series as bin
    from generate_series(0, 46700000, 100000)
),

counts as (
    select
        floor(((start_position + end_position) / 2.0) / 100000) * 100000 as bin,
        count(*) as fragment_count
    from {{ ref('stg_query_fragments') }}
    group by 1
)

select
    samples.sample,
    bins.bin,
    coalesce(counts.fragment_count, 0) as fragment_count
from {{ source('quant', 'quant__samples') }} as samples
cross join bins
left join counts
    on bins.bin = counts.bin
