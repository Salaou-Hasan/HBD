-- Example HBD behaviour script: gentle vertical bob using time and math.
function update(dt)
    hbd.position.y = hbd.position.y + math.sin(hbd.time * 2.0) * dt * 0.5
end
