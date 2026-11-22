-- Example HBD behaviour script: rotate the attached entity around Y.
-- Scripts only see the curated `hbd` table (position/rotation/scale/color,
-- dt, time) plus math/string/table. There is no filesystem or OS access.
function update(dt)
    hbd.rotation.y = hbd.rotation.y + dt * 30.0
end
