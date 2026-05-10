{{ config(tags=['individual']) }}

with distinct_fragments as (
    select distinct
        start_position,
        end_position
    from {{ source('quant', 'quant__records') }}
),

fragment_keys as (
    select
        dense_rank() over (
            order by start_position, end_position
        ) as fragment_id,
        start_position,
        end_position
    from distinct_fragments
)

select
    records.align_id,
    fragment_keys.fragment_id,
    records.start_position,
    records.end_position,
    records.template_length
from {{ source('quant', 'quant__records') }} as records
inner join fragment_keys
    on records.start_position = fragment_keys.start_position
    and records.end_position = fragment_keys.end_position
